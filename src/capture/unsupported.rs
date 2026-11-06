//! Placeholder backend for platforms without a capture implementation yet.
//! The app still builds and runs (background, frames, UI); it just finds no devices.

use super::{Backend, ColorHints, DeviceInfo, Event, FormatInfo, Sink, Stream};

pub struct Unsupported;

pub enum Never {}

impl Backend for Unsupported {
    type Raw = Never;
    type Stream = NoStream;

    fn init() -> Result<Self, String> {
        Ok(Unsupported)
    }
    fn devices(&mut self) -> Vec<DeviceInfo> {
        Vec::new()
    }
    fn open(
        &mut self,
        _link: &str,
        _format_key: Option<&str>,
        _sink: Sink<Never>,
    ) -> Result<(NoStream, Vec<FormatInfo>), String> {
        Err("video capture isn't supported on this OS yet".into())
    }
}

pub struct NoStream(Never);

impl Stream for NoStream {
    type Raw = Never;
    fn selected(&self) -> &FormatInfo {
        match self.0 {}
    }
    fn hints(&self) -> ColorHints {
        match self.0 {}
    }
    fn start(&mut self) -> Result<(), String> {
        match self.0 {}
    }
    fn handle(&mut self, raw: Never, _emit: &mut dyn FnMut(Event<'_>)) -> Result<(), String> {
        match raw {}
    }
}
