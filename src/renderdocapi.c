#include <stddef.h>
#include <assert.h>
#include <stdio.h>

#include <dlfcn.h>

#include <renderdoc_app.h>

void assert_and_complain(int status)
{
    if (!status)
    {
        printf("renderdoc api assertion failed!!\n");
    }

    assert(status);
}

RENDERDOC_API_1_1_2* renderdoc_api_get()
{
    RENDERDOC_API_1_1_2* api = NULL;

    void* renderdoc_lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    assert_and_complain(renderdoc_lib != NULL);

    pRENDERDOC_GetAPI api_getter = (pRENDERDOC_GetAPI)dlsym(renderdoc_lib, "RENDERDOC_GetAPI");

    int code = api_getter(eRENDERDOC_API_Version_1_1_2, (void**)&api);
    assert_and_complain(code == 1);

    assert_and_complain(api != NULL);

    api->UnloadCrashHandler();

    api->SetCaptureFilePathTemplate("./capture");

    printf("captures saved in: %s\n", api->GetCaptureFilePathTemplate());

    return api;
}

void renderdoc_api_begin_capture(RENDERDOC_API_1_1_2* api)
{
    printf("renderdoc began capture\n");

    api->StartFrameCapture(NULL, NULL);
}

void renderdoc_api_end_capture(RENDERDOC_API_1_1_2* api)
{
    printf("renderdoc ended capture\n");

    api->EndFrameCapture(NULL, NULL);
}
