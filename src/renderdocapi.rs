#[cfg(feature = "renderdocapi")]
#[link(name = "renderdocapi")]
unsafe extern "C"
{
    pub safe fn renderdoc_api_get() -> *mut core::ffi::c_void;

    pub unsafe fn renderdoc_api_begin_capture(api: *mut core::ffi::c_void);
    pub unsafe fn renderdoc_api_end_capture(api: *mut core::ffi::c_void);
}

#[cfg(feature = "renderdocapi")]
pub struct RenderDocApi(*mut core::ffi::c_void);

#[cfg(feature = "renderdocapi")]
unsafe impl Send for RenderDocApi {}

#[cfg(feature = "renderdocapi")]
unsafe impl Sync for RenderDocApi {}

#[cfg(feature = "renderdocapi")]
impl RenderDocApi
{
    fn init() -> Self
    {
        Self(renderdoc_api_get())
    }

    pub fn begin_capture(&self)
    {
        unsafe{ renderdoc_api_begin_capture(self.0); }
    }

    pub fn end_capture(&self)
    {
        unsafe{ renderdoc_api_end_capture(self.0); }
    }
}

#[cfg(feature = "renderdocapi")]
pub static RENDERDOC_API: std::sync::LazyLock<RenderDocApi> = std::sync::LazyLock::new(|| RenderDocApi::init());
