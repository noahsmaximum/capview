//! NVIDIA RTX Video Super Resolution (NGX) running on wgpu's own D3D12 device and
//! queue. The upscale is recorded into our own command list and executed on wgpu's
//! queue *between* two wgpu submissions, so ordering comes free from the queue.
//!
//! Resource states: wgpu (legacy barriers) believes both textures are render targets
//! when we run: the input was just drawn by the prep pass and the output is "touched"
//! by an empty render pass in the same submission. We move both to COMMON (what the
//! SDK samples hand NGX), evaluate, and put them back to RENDER_TARGET, so wgpu's
//! tracking stays true.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

type Dx12 = wgpu::hal::api::Dx12;

unsafe extern "C" {
    fn cv_ngx_init(device: *mut c_void, data_path: *const u16) -> i32;
    fn cv_ngx_vsr_caps(
        available: *mut i32,
        needs_driver: *mut i32,
        min_major: *mut u32,
        min_minor: *mut u32,
    ) -> i32;
    fn cv_ngx_vsr_create(list: *mut c_void, out: *mut *mut c_void) -> i32;
    fn cv_ngx_vsr_eval(
        list: *mut c_void,
        handle: *mut c_void,
        input: *mut c_void,
        in_rect: *const u32,
        output: *mut c_void,
        out_rect: *const u32,
        quality: i32,
    ) -> i32;
    fn cv_ngx_vsr_release(handle: *mut c_void);
    fn cv_ngx_shutdown(device: *mut c_void);
}

fn ngx_ok(r: i32) -> bool {
    (r as u32 & 0xFFF0_0000) != 0xBAD0_0000
}

const SLOTS: usize = 3;

struct Slot {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Fence value that marks this slot's last submission as finished.
    done_at: u64,
    timed: bool,
}

pub struct Vsr {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    handle: *mut c_void,
    slots: Vec<Slot>,
    next: usize,
    fence: ID3D12Fence,
    fence_value: u64,
    event: HANDLE,
    queries: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ts_freq: f64,
    /// GPU time of the most recently completed upscale.
    pub gpu_ms: Option<f32>,
}

impl Vsr {
    /// Initialise NGX on wgpu's device. Errors are user-facing reasons.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Self, String> {
        let raw_device = unsafe { device.as_hal::<Dx12>() }
            .map(|d| d.raw_device().clone())
            .ok_or("needs the DirectX 12 renderer")?;
        let raw_queue = unsafe { queue.as_hal::<Dx12>() }
            .map(|q| q.as_raw().clone())
            .ok_or("needs the DirectX 12 renderer")?;

        let data_dir = crate::settings::config_dir();
        let _ = std::fs::create_dir_all(&data_dir);
        let wide: Vec<u16> = data_dir
            .as_os_str()
            .to_string_lossy()
            .encode_utf16()
            .chain(Some(0))
            .collect();
        let r = unsafe { cv_ngx_init(raw_device.as_raw(), wide.as_ptr()) };
        if !ngx_ok(r) {
            return Err(format!(
                "needs an NVIDIA RTX GPU and driver (NGX init 0x{:08X})",
                r as u32
            ));
        }
        let (mut avail, mut needs, mut maj, mut min) = (0, 0, 0u32, 0u32);
        unsafe { cv_ngx_vsr_caps(&mut avail, &mut needs, &mut maj, &mut min) };
        if avail == 0 {
            unsafe { cv_ngx_shutdown(raw_device.as_raw()) };
            return Err(if needs != 0 {
                format!("update the NVIDIA driver (needs {maj}.{min} or newer)")
            } else {
                "not supported on this GPU (needs GeForce RTX)".into()
            });
        }

        let mut vsr = unsafe { Self::create_objects(raw_device, raw_queue) }
            .map_err(|e| format!("D3D12 setup failed: {e}"))?;

        // Create the feature on its own list, then wait so the first evaluate is safe.
        let slot = &vsr.slots[0];
        let r = unsafe {
            slot.alloc.Reset().map_err(|e| e.to_string())?;
            slot.list
                .Reset(&slot.alloc, None)
                .map_err(|e| e.to_string())?;
            let r = cv_ngx_vsr_create(slot.list.as_raw(), &mut vsr.handle);
            slot.list.Close().map_err(|e| e.to_string())?;
            r
        };
        if !ngx_ok(r) || vsr.handle.is_null() {
            return Err(format!("couldn't create VSR (0x{:08X})", r as u32));
        }
        vsr.submit(0, false);
        vsr.wait_idle();
        Ok(vsr)
    }

    unsafe fn create_objects(
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
    ) -> windows::core::Result<Self> {
        unsafe {
            let mut slots = Vec::with_capacity(SLOTS);
            for _ in 0..SLOTS {
                let alloc: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)?;
                list.Close()?;
                slots.push(Slot {
                    alloc,
                    list,
                    done_at: 0,
                    timed: false,
                });
            }
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let mut queries: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: (SLOTS * 2) as u32,
                    NodeMask: 0,
                },
                &mut queries,
            )?;
            let mut readback: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: (SLOTS * 2 * 8) as u64,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )?;
            let ts_freq = queue.GetTimestampFrequency().unwrap_or(0) as f64;
            Ok(Self {
                device,
                queue,
                handle: std::ptr::null_mut(),
                slots,
                next: 0,
                fence,
                fence_value: 0,
                event,
                queries: queries.unwrap(),
                readback: readback.unwrap(),
                ts_freq,
                gpu_ms: None,
            })
        }
    }

    fn submit(&mut self, slot: usize, timed: bool) {
        let s = &mut self.slots[slot];
        let list: ID3D12CommandList = s.list.cast().expect("command list");
        self.fence_value += 1;
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(list)]);
            let _ = self.queue.Signal(&self.fence, self.fence_value);
        }
        s.done_at = self.fence_value;
        s.timed = timed;
    }

    fn wait_for(&self, value: u64) {
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            if self.fence.SetEventOnCompletion(value, self.event).is_ok() {
                WaitForSingleObject(self.event, INFINITE);
            }
        }
    }

    /// Block until every upscale we've submitted has finished on the GPU. Call before
    /// dropping textures that an upscale may still be reading or writing (wgpu doesn't
    /// know about our submissions).
    pub fn wait_idle(&self) {
        self.wait_for(self.fence_value);
    }

    /// Record and submit one upscale of `input[in_rect]` into `output[out_rect]`
    /// (rects are x, y, w, h). Must run right after the wgpu submission that drew
    /// `input` and touched `output` as render targets, and before the one that samples
    /// `output`. Quality 1..=4.
    pub fn evaluate(
        &mut self,
        input: &wgpu::Texture,
        in_rect: [u32; 4],
        output: &wgpu::Texture,
        out_rect: [u32; 4],
        quality: i32,
    ) -> Result<(), String> {
        let (Some(inp), Some(out)) = (
            unsafe { input.as_hal::<Dx12>() }.map(|t| unsafe { t.raw_resource().clone() }),
            unsafe { output.as_hal::<Dx12>() }.map(|t| unsafe { t.raw_resource().clone() }),
        ) else {
            return Err("not a D3D12 texture".into());
        };

        let i = self.next;
        self.next = (self.next + 1) % SLOTS;
        self.wait_for(self.slots[i].done_at);
        if self.slots[i].timed {
            self.read_timing(i);
        }
        let s = &self.slots[i];
        let q = (i * 2) as u32;
        let r = unsafe {
            s.alloc.Reset().map_err(|e| e.to_string())?;
            s.list.Reset(&s.alloc, None).map_err(|e| e.to_string())?;
            let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
            let common = D3D12_RESOURCE_STATE_COMMON;
            transition(&s.list, &[(&inp, rt, common), (&out, rt, common)]);
            s.list
                .EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, q);
            let r = cv_ngx_vsr_eval(
                s.list.as_raw(),
                self.handle,
                inp.as_raw(),
                in_rect.as_ptr(),
                out.as_raw(),
                out_rect.as_ptr(),
                quality.clamp(1, 4),
            );
            s.list
                .EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, q + 1);
            s.list.ResolveQueryData(
                &self.queries,
                D3D12_QUERY_TYPE_TIMESTAMP,
                q,
                2,
                &self.readback,
                q as u64 * 8,
            );
            transition(&s.list, &[(&inp, common, rt), (&out, common, rt)]);
            s.list.Close().map_err(|e| e.to_string())?;
            r
        };
        // Submit even on failure: the barriers must still run to keep wgpu's state
        // tracking true.
        self.submit(i, ngx_ok(r));
        if ngx_ok(r) {
            Ok(())
        } else {
            Err(format!("VSR evaluate failed (0x{:08X})", r as u32))
        }
    }

    fn read_timing(&mut self, slot: usize) {
        if self.ts_freq <= 0.0 {
            return;
        }
        let range = D3D12_RANGE {
            Begin: slot * 16,
            End: slot * 16 + 16,
        };
        unsafe {
            let mut p: *mut c_void = std::ptr::null_mut();
            if self.readback.Map(0, Some(&range), Some(&mut p)).is_ok() && !p.is_null() {
                let ts = std::slice::from_raw_parts((p as *const u8).add(slot * 16) as *const u64, 2);
                if ts[1] > ts[0] {
                    let ms = ((ts[1] - ts[0]) as f64 / self.ts_freq * 1000.0) as f32;
                    self.gpu_ms = Some(match self.gpu_ms {
                        Some(prev) => prev * 0.9 + ms * 0.1,
                        None => ms,
                    });
                }
                self.readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            }
        }
    }
}

/// Transition whole resources. Borrows the interfaces without touching refcounts.
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

impl Drop for Vsr {
    fn drop(&mut self) {
        self.wait_idle();
        unsafe {
            cv_ngx_vsr_release(self.handle);
            cv_ngx_shutdown(self.device.as_raw());
            let _ = CloseHandle(self.event);
        }
    }
}
