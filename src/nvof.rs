//! NVIDIA Optical Flow (NVOFA hardware) on wgpu's own D3D12 device: forward and
//! backward motion vectors between two luma frames, for capview's frame generation.
//!
//! `nvofapi64.dll` ships with the NVIDIA driver, so nothing from the SDK is
//! compiled in or redistributed. The engine runs on its own queue and syncs with
//! ours through a D3D12 fence. Every texture it touches is a wgpu texture: wgpu
//! believes they're render targets (the frame-gen code "touches" the flow textures
//! and draws the luma ones), so our lists move them RT → COMMON before the engine
//! reads/writes them and COMMON → RT afterwards, keeping wgpu's tracking true.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, w};

type Dx12 = wgpu::hal::api::Dx12;
type OfHandle = *mut c_void;
type BufHandle = *mut c_void;

const API_VERSION: u32 = (5 << 4) | 0;
const MODE_OPTICALFLOW: i32 = 1;
const PERF_MEDIUM: i32 = 10;
const FORMAT_GRAYSCALE8: i32 = 1;
const PRED_BOTH: i32 = 2;
const CAPS_OUTPUT_GRID_SIZES: i32 = 0;

#[repr(C)]
struct InitParams {
    width: u32,
    height: u32,
    out_grid: i32,
    hint_grid: i32,
    mode: i32,
    perf: i32,
    external_hints: i32,
    output_cost: i32,
    priv_data: *mut c_void,
    disparity_range: i32,
    roi: i32,
    pred_direction: i32,
    global_flow: i32,
    input_format: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FencePoint {
    fence: *mut c_void,
    value: u64,
}

#[repr(C)]
struct RegisterParams {
    resource: *mut c_void,
    input_fence: FencePoint,
    handle: *mut BufHandle,
    output_fence: FencePoint,
}

#[repr(C)]
struct UnregisterParams {
    handle: BufHandle,
}

#[repr(C)]
struct ExecIn {
    input: BufHandle,
    reference: BufHandle,
    hints: BufHandle,
    disable_temporal_hints: i32,
    padding: u32,
    priv_data: *mut c_void,
    padding2: u32,
    num_rois: u32,
    rois: *mut c_void,
    padding3: u32,
    num_fences: u32,
    fences: *const FencePoint,
}

#[repr(C)]
struct ExecOut {
    output: BufHandle,
    cost: BufHandle,
    priv_data: *mut c_void,
    bwd_output: BufHandle,
    bwd_cost: BufHandle,
    global_flow: BufHandle,
    padding: [u32; 2],
    fence: *const FencePoint,
}

#[repr(C)]
#[derive(Default)]
struct Api {
    create: Option<unsafe extern "system" fn(*mut c_void, *mut OfHandle) -> i32>,
    init: Option<unsafe extern "system" fn(OfHandle, *const InitParams) -> i32>,
    format_count: Option<unsafe extern "system" fn(OfHandle, i32, i32, *mut u32) -> i32>,
    format: Option<unsafe extern "system" fn(OfHandle, i32, i32, *mut i32) -> i32>,
    register: Option<unsafe extern "system" fn(OfHandle, *mut RegisterParams) -> i32>,
    unregister: Option<unsafe extern "system" fn(*mut UnregisterParams) -> i32>,
    execute: Option<unsafe extern "system" fn(OfHandle, *const ExecIn, *mut ExecOut) -> i32>,
    destroy: Option<unsafe extern "system" fn(OfHandle) -> i32>,
    last_error: Option<unsafe extern "system" fn(OfHandle, *mut u8, *mut u32) -> i32>,
    caps: Option<unsafe extern "system" fn(OfHandle, i32, *mut u32, *mut u32) -> i32>,
}

fn status_text(s: i32) -> &'static str {
    match s {
        1 => "optical flow not available",
        2 => "GPU has no optical flow engine (needs Turing or newer)",
        5 => "invalid parameter",
        6 => "invalid call",
        7 => "driver too old for Optical Flow 5.0",
        8 => "out of memory",
        10 => "unsupported feature",
        _ => "error",
    }
}

/// The textures one flow pass works on (all wgpu render targets between uses).
pub struct FlowSet<'a> {
    pub prev_luma: &'a wgpu::Texture,
    pub next_luma: &'a wgpu::Texture,
    /// prev → next motion, one vector per `grid`×`grid` block (Rg16Sint, S10.5 px).
    pub forward: &'a wgpu::Texture,
    /// next → prev motion.
    pub backward: &'a wgpu::Texture,
}

pub struct OpticalFlow {
    api: Api,
    of: OfHandle,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    event: HANDLE,
    alloc: [ID3D12CommandAllocator; 4],
    lists: [ID3D12GraphicsCommandList; 4],
    list_done: [u64; 4],
    next_list: usize,
    /// Registered resources by their raw pointer.
    registered: Vec<(*mut c_void, BufHandle, ID3D12Resource)>,
    pub grid: u32,
    pub size: [u32; 2],
}

impl OpticalFlow {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> Result<Self, String> {
        let d12 = unsafe { device.as_hal::<Dx12>() }
            .map(|d| d.raw_device().clone())
            .ok_or("needs the DirectX 12 renderer")?;
        let q12 = unsafe { queue.as_hal::<Dx12>() }
            .map(|q| q.as_raw().clone())
            .ok_or("needs the DirectX 12 renderer")?;
        let module = unsafe { LoadLibraryW(w!("nvofapi64.dll")) }
            .map_err(|_| "needs an NVIDIA GPU and driver (nvofapi64.dll not found)".to_string())?;
        let create_instance = unsafe { GetProcAddress(module, PCSTR(b"NvOFAPICreateInstanceD3D12\0".as_ptr())) }
            .ok_or("driver has no D3D12 optical flow")?;
        let create_instance: unsafe extern "system" fn(u32, *mut Api) -> i32 =
            unsafe { std::mem::transmute(create_instance) };
        let mut api = Api::default();
        let st = unsafe { create_instance(API_VERSION, &mut api) };
        if st != 0 {
            return Err(format!("optical flow: {} ({st})", status_text(st)));
        }
        let (Some(create), Some(init), Some(caps)) = (api.create, api.init, api.caps) else {
            return Err("optical flow API incomplete".into());
        };
        let mut of: OfHandle = std::ptr::null_mut();
        let st = unsafe { create(d12.as_raw(), &mut of) };
        if st != 0 || of.is_null() {
            return Err(format!("optical flow: {} ({st})", status_text(st)));
        }

        // Finest supported grid of 1 or 2 px (1 px vectors cost more; 2 is plenty
        // for interpolation and matches what the warp shader expects to sample).
        let mut grids = [0u32; 8];
        let mut n = grids.len() as u32;
        let grid = if unsafe { caps(of, CAPS_OUTPUT_GRID_SIZES, grids.as_mut_ptr(), &mut n) } == 0
            && grids[..n as usize].contains(&2)
        {
            2
        } else {
            4
        };

        let ip = InitParams {
            width: w,
            height: h,
            out_grid: grid as i32,
            hint_grid: 0,
            mode: MODE_OPTICALFLOW,
            perf: PERF_MEDIUM,
            external_hints: 0,
            output_cost: 0,
            priv_data: std::ptr::null_mut(),
            disparity_range: 0,
            roi: 0,
            pred_direction: PRED_BOTH,
            global_flow: 0,
            input_format: FORMAT_GRAYSCALE8,
        };
        let st = unsafe { init(of, &ip) };
        if st != 0 {
            let msg = Self::error_text(&api, of);
            if let Some(d) = api.destroy {
                unsafe { d(of) };
            }
            return Err(format!("optical flow init: {} ({st}) {msg}", status_text(st)));
        }

        let setup = |e: windows::core::Error| format!("D3D12 setup failed: {e}");
        unsafe {
            let fence: ID3D12Fence = d12.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(setup)?;
            let event = CreateEventW(None, false, false, None).map_err(setup)?;
            let mk = || -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
                let a: ID3D12CommandAllocator = d12.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let l: ID3D12GraphicsCommandList =
                    d12.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)?;
                l.Close()?;
                Ok((a, l))
            };
            let (a0, l0) = mk().map_err(setup)?;
            let (a1, l1) = mk().map_err(setup)?;
            let (a2, l2) = mk().map_err(setup)?;
            let (a3, l3) = mk().map_err(setup)?;
            Ok(Self {
                api,
                of,
                queue: q12,
                fence,
                value: 0,
                event,
                alloc: [a0, a1, a2, a3],
                lists: [l0, l1, l2, l3],
                list_done: [0; 4],
                next_list: 0,
                registered: Vec::new(),
                grid,
                size: [w, h],
            })
        }
    }

    fn error_text(api: &Api, of: OfHandle) -> String {
        let Some(f) = api.last_error else { return String::new() };
        let mut buf = [0u8; 256];
        let mut n = buf.len() as u32;
        unsafe { f(of, buf.as_mut_ptr(), &mut n) };
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).trim().to_string()
    }

    fn fence_point(&self, value: u64) -> FencePoint {
        FencePoint {
            fence: self.fence.as_raw(),
            value,
        }
    }

    fn wait_cpu(&self, value: u64) {
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            if self.fence.SetEventOnCompletion(value, self.event).is_ok() {
                WaitForSingleObject(self.event, INFINITE);
            }
        }
    }

    pub fn wait_idle(&self) {
        self.wait_cpu(self.value);
    }

    /// Handle for a texture, registering it on first use.
    fn handle(&mut self, tex: &wgpu::Texture) -> Result<(BufHandle, ID3D12Resource), String> {
        let res = unsafe { tex.as_hal::<Dx12>() }
            .map(|t| unsafe { t.raw_resource().clone() })
            .ok_or("not a D3D12 texture")?;
        let key = res.as_raw();
        if let Some((_, h, r)) = self.registered.iter().find(|(k, _, _)| *k == key) {
            return Ok((*h, r.clone()));
        }
        let register = self.api.register.ok_or("optical flow API incomplete")?;
        let mut h: BufHandle = std::ptr::null_mut();
        // Registration is a GPU op too: give it an already-reached wait point and a
        // fresh signal point.
        let wait = self.fence_point(self.value);
        self.value += 1;
        let done = self.fence_point(self.value);
        let mut p = RegisterParams {
            resource: key,
            input_fence: wait,
            handle: &mut h,
            output_fence: done,
        };
        let st = unsafe { register(self.of, &mut p) };
        if st != 0 || h.is_null() {
            return Err(format!(
                "optical flow register: {} ({st}) {}",
                status_text(st),
                Self::error_text(&self.api, self.of)
            ));
        }
        self.wait_cpu(self.value);
        self.registered.push((key, h, res.clone()));
        Ok((h, res))
    }

    /// Forget registrations of textures that are about to be dropped.
    pub fn unregister_all(&mut self) {
        self.wait_idle();
        if let Some(u) = self.api.unregister {
            for (_, h, _) in self.registered.drain(..) {
                let mut p = UnregisterParams { handle: h };
                unsafe { u(&mut p) };
            }
        }
    }

    fn run_list(
        &mut self,
        items: &[(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)],
    ) -> Result<(), String> {
        let i = self.next_list;
        self.next_list = (i + 1) % self.lists.len();
        self.wait_cpu(self.list_done[i]);
        unsafe {
            let (a, l) = (&self.alloc[i], &self.lists[i]);
            a.Reset().map_err(|e| e.to_string())?;
            l.Reset(a, None).map_err(|e| e.to_string())?;
            transition(l, items);
            l.Close().map_err(|e| e.to_string())?;
            let cl: ID3D12CommandList = l.cast().map_err(|e| e.to_string())?;
            self.queue.ExecuteCommandLists(&[Some(cl)]);
        }
        self.value += 1;
        unsafe {
            self.queue
                .Signal(&self.fence, self.value)
                .map_err(|e| e.to_string())?;
        }
        self.list_done[i] = self.value;
        Ok(())
    }

    /// Queue one bidirectional flow pass. Call after the wgpu submission that wrote
    /// both luma frames and touched both flow textures; anything submitted to the
    /// queue afterwards sees the finished vectors.
    pub fn execute(&mut self, set: FlowSet) -> Result<(), String> {
        let (h_prev, r_prev) = self.handle(set.prev_luma)?;
        let (h_next, r_next) = self.handle(set.next_luma)?;
        let (h_fwd, r_fwd) = self.handle(set.forward)?;
        let (h_bwd, r_bwd) = self.handle(set.backward)?;
        let execute = self.api.execute.ok_or("optical flow API incomplete")?;
        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let common = D3D12_RESOURCE_STATE_COMMON;
        let all = [&r_prev, &r_next, &r_fwd, &r_bwd];

        let to_common: Vec<_> = all.iter().map(|r| (*r, rt, common)).collect();
        self.run_list(&to_common)?;
        let ready = self.fence_point(self.value);
        self.value += 1;
        let done = self.fence_point(self.value);
        // The engine's "input frame" is the one vectors start from: prev → next.
        let input = ExecIn {
            input: h_prev,
            reference: h_next,
            hints: std::ptr::null_mut(),
            disable_temporal_hints: 0,
            padding: 0,
            priv_data: std::ptr::null_mut(),
            padding2: 0,
            num_rois: 0,
            rois: std::ptr::null_mut(),
            padding3: 0,
            num_fences: 1,
            fences: &ready,
        };
        let mut output = ExecOut {
            output: h_fwd,
            cost: std::ptr::null_mut(),
            priv_data: std::ptr::null_mut(),
            bwd_output: h_bwd,
            bwd_cost: std::ptr::null_mut(),
            global_flow: std::ptr::null_mut(),
            padding: [0; 2],
            fence: &done,
        };
        let st = unsafe { execute(self.of, &input, &mut output) };
        if st != 0 {
            // Keep the fence and wgpu's state tracking consistent even on failure.
            unsafe {
                let _ = self.queue.Signal(&self.fence, done.value);
            }
            let back: Vec<_> = all.iter().map(|r| (*r, common, rt)).collect();
            self.run_list(&back)?;
            return Err(format!(
                "optical flow: {} ({st}) {}",
                status_text(st),
                Self::error_text(&self.api, self.of)
            ));
        }
        unsafe {
            self.queue
                .Wait(&self.fence, done.value)
                .map_err(|e| e.to_string())?;
        }
        let back: Vec<_> = all.iter().map(|r| (*r, common, rt)).collect();
        self.run_list(&back)
    }
}

unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
    items: &[(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)],
) {
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = items
        .iter()
        .map(|(res, before, after)| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(*res) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: *before,
                    StateAfter: *after,
                }),
            },
        })
        .collect();
    unsafe { list.ResourceBarrier(&barriers) };
}

impl Drop for OpticalFlow {
    fn drop(&mut self) {
        self.unregister_all();
        if let Some(d) = self.api.destroy {
            unsafe { d(self.of) };
        }
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}
