# OCR

https://github.com/thebesttv/PonyOCR

https://github.com/louis845/COMP2012H_Project

https://github.com/IshiKura-a/CPP_PROJECT_2020

https://github.com/itewqq/MathpixCsharp

https://github.com/xingchenzhao/MathFlow



# Pan

http://vip.fulivip.com/lin/GI5LG4?refer=1107



https://github.com/wuhuikai/PointCloudSuperResolution

https://github.com/krrish94/chamferdist

https://github.com/daerduoCarey/PyTorchEMD



cuda

https://github.com/praeclarumjj3/CuML

https://cloud.tencent.com/developer/article/1175256

https://github.com/depctg/udacity-cs344-colab

https://github.com/udacity/cs344



https://github.com/AnTao97/PointCloudDatasets

https://github.com/QingyongHu/SoTA-Point-Cloud



javascirpt可视化

https://github.com/verma/plasio

https://github.com/potree/PotreeConverter

https://github.com/potree/potree

https://github.com/tentone/potree-core

https://github.com/francisengelmann/PyViz3D





# Apex

## How to make a Box ESP for Apex Legends (C++)

https://www.unknowncheats.me/forum/apex-legends/445786-box-esp-apex-legends.html



Hello everyone, I thought that I would create a guide for making a box esp for apex, since there's a lack of sources, and I thought an explanation would be good.

Here is a screenshot of what you should end up with at the end of this tutorial: https://media.discordapp.net/attachm...58/ApexESP.png

First, lets define a few offsets. (These will most likely change with every update, so these will most likely not be up-to-date by the time you are reading this)

```c++
#define OFFSET_ENTITYLIST 0x18DA338
#define OFFSET_MATRIX 0x1B3BD0
#define OFFSET_RENDER 0x408B768
#define OFFSET_ORIGIN 0x14C
```

Second, there are two structs we need to set up. Vector3, and Matrix.

Vector3 is used to store 3 floats, the x, y, and z coordinates.

```c++
struct Vector3 {	
    float x, y, z;
};
```

Matrix is used to store 16 bytes for our view matrix.

```c++
struct Matrix {	
    float matrix[16];
};
```

Now we will create our WorldToScreen struct. This converts World Space to Screen Space (in-game coordinates: x, y, z to Screen Coordinates: x, y)

```c++
struct Vector3 _WorldToScreen(const struct Vector3 pos, struct Matrix matrix) {
	struct Vector3 out;
	float _x = matrix.matrix[0] * pos.x + matrix.matrix[1] * pos.y + matrix.matrix[2] * pos.z + matrix.matrix[3];
	float _y = matrix.matrix[4] * pos.x + matrix.matrix[5] * pos.y + matrix.matrix[6] * pos.z + matrix.matrix[7];
	out.z = matrix.matrix[12] * pos.x + matrix.matrix[13] * pos.y + matrix.matrix[14] * pos.z + matrix.matrix[15];
 
	_x *= 1.f / out.z;
	_y *= 1.f / out.z;
 
	int width = 1920; //Change this to your resolution.
	int height = 1080;
 
	out.x = width * .5f;
	out.y = height * .5f;
 
	out.x += 0.5f * _x * width + 0.5f;
	out.y -= 0.5f * _y * height + 0.5f;
 
	return out;
}
```



Next we will create our `getEntityById` function, this way we will be able to loop through every entity.

```c++
DWORD64 GetEntityById(int Ent, DWORD64 Base) {
	DWORD64 EntityList = Base + OFFSET_ENTITYLIST;
	DWORD64 BaseEntity = Read<DWORD64>(EntityList);
	if (!BaseEntity) 
        return NULL;
	return  Read<DWORD64>(EntityList + (Ent << 5));
}
```



Now we can move onto our main function. This tutorial assumes that you have your **process id**, **base address**, and **read/write memory functions**, so if you don't have that, write it, then come back.

Inside your main function, you should have an **infinite loop**, where you can call your cheat functions. Inside of this loop, we will read the view render+view matrix, iterate through every entity, retrieve their position, etc.

The first step in our main function is to get our view matrix set up. For this, we will be **read**ing the viewRender, **read**ing the viewMatrix and **add**ing viewRender to it, and then use that Matrix struct we defined earlier to create a variable to store our viewMatrix.

```c++
while (true) {
    uint64_t viewRenderer = Read<uint64_t>(base_address + OFFSET_RENDER);
    uint64_t viewMatrix = Read<uint64_t>(viewRenderer + OFFSET_MATRIX);
    Matrix m = Read<Matrix>(viewMatrix);
}
```

After our view matrix, we will **iterate through every entity**. (This is directly below the previous code block, in the main function and in the infinite loop)

Once we have gone through every entity, we will find their position using the origin offset (which is the player's **feet** position) and then from there we will roughly find the position of their **head** by adding **35** to the z coordinate(you can use bones, but for this tutorial, I wont be doing that). Once we have the head and feet Vector3's, we can create two more Vector3's that use the world to screen function, inputting entFeet or entHead as our pos parameter, and then m (the matrix variable we made earlier that uses the matrix struct) as our matrix.

```c++
for (int i = 0; i < 100; i++) {
	DWORD64 Entity = GetEntityById(i, base_address);
	if (Entity == 0)
		continue;
 
    Vector3 entFeet = Read<Vector3>(Entity + OFFSET_ORIGIN);
	Vector3 entHead = entFeet; entHead.z += 35.f;
	Vector3 w2sEntFeet = _WorldToScreen(entFeet, m); if (w2sEntFeet.z <= 0.f) continue;
	Vector3 w2sEntHead = _WorldToScreen(entHead, m); if (w2sEntHead.z <= 0.f) continue;
}
```

Now for the exciting part, rendering our ESP.  （可以，但是每次都调用RPM来获取实体列表吗？不要那样做。它将使您的作弊速度变慢。尽可能少的rpm调用，如果可能，读取一个数组，然后将其拆分。）

The next part is **crucial**: If you do not have a way to render a box in your game, or an **overlay**, I will be going over this using the NVIDIA Overlay Hijacker:  [https://github.com/iraizo/nvidia-overlay-hijack/tree/master/src/overlay](https://github.com/iraizo/nvidia-overlay-hijack/tree/master/src/overlay)



We will be creating a new function for the esp, **inputting coordinates of our head and feet Vector3** into whatever draw box function you have, to make the X, Y, Width, and Height, parameters.

This Code block is for those who **already have a render box function** `DrawBox()` that takes in a x, y, width, height (maybe thickness) parameters.  

```c++
void DrawBoxESP(Vector3 foot, Vector3 head) {
	float height = head.y - foot.y;
	float width = height / 1.2f;
 
    //2.0f is the box thickness
	DrawBox(foot.x - (width / 2), foot.y, head.x + width, head.y + height, 2.0f); 
}
```



After you have your esp function done, go into your entity loop right after our `w2sEntHead`, and simply call the function:

```c++
DrawBoxESP(w2sEntFeet, w2sEntHead);
```

And that's it!

```c++
for (int i = 0; i < 100; i++) {
	DWORD64 Entity = GetEntityById(i, base_address);
	if (Entity == 0)
		continue;
 
    Vector3 entFeet = Read<Vector3>(Entity + OFFSET_ORIGIN);
	Vector3 entHead = entFeet; entHead.z += 35.f;
	Vector3 w2sEntFeet = _WorldToScreen(entFeet, m); if (w2sEntFeet.z <= 0.f) continue;
	Vector3 w2sEntHead = _WorldToScreen(entHead, m); if (w2sEntHead.z <= 0.f) continue;
	DrawBoxESP(w2sEntFeet, w2sEntHead);
}
```



Now for people who want to use the `nvidia-overlay-hijack`. Put the `FOverlay.cpp` and `FOverlay.h` into your project. Then you need to go into `FOverlay.cpp` and **add this function** (near the `draw_text_white` function around line 130.):

```c++
auto FOverlay::draw_box(int x, int y, int width, int height, float thickness, ...)-> void {
	tar->DrawRectangle(D2D1::RectF(x, y, width, height), red_brush, thickness);
}
```

And now **put the function** in your `FOverlay.h` (line 36):

```c++
auto draw_box(int x, int y, int width, int height, float thickness, ...)-> void;
```



Once you have that done and saved, go back to your main file (make sure you included `FOverlay.h`), and now we will create the `ESP function`:

```C++
void DrawBoxESP(FOverlay* overlay, Vector3 foot, Vector3 head)
{
	float height = head.y - foot.y;
	float width = height / 1.2f;
	overlay->draw_box(foot.x - (width / 2), foot.y, head.x + width, head.y + height, 2.0f); //ESP BOX
}
```



Then we can simply call our ESP function in our entity loop:

```c++
DrawBoxESP(overlay, w2sEntFeet, w2sEntHead);
```

If you have **Nvidia overlay initialized correctly**, you should have a working esp.

What the main function should look like if you are using nvidia overlay (I also included a **kill key**, so that you can take off the overlay:  

```c++
	FOverlay* overlay = { 0 };
	overlay->window_init();
	overlay->init_d2d();
 
	while (true) {
		if (GetAsyncKeyState(VK_END)) {
			overlay->begin_scene();
 
			overlay->clear_scene();
 
			overlay->end_scene();
 
			overlay->d2d_shutdown();
		}
 
		uint64_t viewRenderer = Read<uint64_t>(base_address + OFFSET_RENDER);
		uint64_t viewMatrix = Read<uint64_t>(viewRenderer + OFFSET_MATRIX);
		Matrix m = Read<Matrix>(viewMatrix);
 
		overlay->begin_scene();
 
		overlay->clear_scene();
 
		for (int i = 0; i < 100; i++) {
			DWORD64 Entity = GetEntityById(i, base_address);
			if (Entity == 0)
				continue;
 
			Vector3 entFeet = Read<Vector3>(Entity + OFFSET_ORIGIN);
			Vector3 entHead = entFeet; entHead.z += 35.f;
			Vector3 w2sEntFeet = _WorldToScreen(entFeet, m); if (w2sEntFeet.z <= 0.f) continue;
			Vector3 w2sEntHead = _WorldToScreen(entHead, m); if (w2sEntHead.z <= 0.f) continue;
 
			DrawBoxESP(overlay, w2sEntFeet, w2sEntHead);
		}
 
		overlay->end_scene();
	}
```





**找基址**

1) You need to change your Character set to Multi-Byte (look up how to do it)
2) Make a header file name what ever you want (make sure to include it in your main.cpp)
3) run the following code:

```c++
#pragma once
#include <iostream>
#include <Windows.h>
#include <TlHelp32.h>
 
using namespace std;
 
DWORD pid = 0;
 
DWORD GetProcessID(const char* process)
{
	PROCESSENTRY32 processInfo;
	processInfo.dwSize = sizeof(processInfo);
 
 
	HANDLE processesSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, NULL);
	if (processesSnapshot == INVALID_HANDLE_VALUE)
		return NULL;
 
	Process32First(processesSnapshot, &processInfo);
	if (!strcmp(processInfo.szExeFile, process))
	{
		CloseHandle(processesSnapshot);
	}
 
	while (Process32Next(processesSnapshot, &processInfo))
	{
		if (!strcmp(processInfo.szExeFile, process))
		{
			CloseHandle(processesSnapshot);
		}
	}
	return processInfo.th32ProcessID;
}
 
uintptr_t baseAddy = 0;
uintptr_t getBaseAddress(const char* process, DWORD pid)
{
	HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid);
	if (hSnap != INVALID_HANDLE_VALUE)
	{
		MODULEENTRY32 modEntry;
		modEntry.dwSize = sizeof(modEntry);
		if (Module32First(hSnap, &modEntry))
		{
			do
			{
				if (!strcmp(modEntry.szModule, process))
				{
					CloseHandle(hSnap);
					return (uintptr_t)modEntry.modBaseAddr;
				}
			} while (Module32Next(hSnap, &modEntry));
		}
	}
}
 
HANDLE hprocess = 0;
 
template <typename T>
T Read(uintptr_t address)
{
	T buffer;
	ReadProcessMemory(hprocess, (LPVOID)address, &buffer, sizeof(buffer), NULL);
	return buffer;
}
 
template <typename T>
void Write(uintptr_t address, T value)
{
	WriteProcessMemory(hprocess, (LPVOID)address, &value, sizeof(value), NULL);
}
 
void attachToProc()
{
	pid = GetProcessID("r5apex.exe");
	baseAddy = getBaseAddress("r5apex.exe", pid);
	hprocess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
}
```





## **Inject dll in Apex and Internal wallhack**

https://www.unknowncheats.me/forum/apex-legends/387569-inject-dll-apex.html


Edit: Injector still work but Internal p200 need an update so Apex will just crash
I sow that peaple don't know how to inject a dll in apex so I made this thread to explain how to do so

I tried many dll injectors but the only one that work for me is modmap: https://github.com/btbd/modmap

-To use it you need to load the `driver.sys` with **kdmapper** to do so open cmd then type，`cd "the folder of your dll, modmap and kdmapper"` 。For example: 

```shell
cd C:\Users\Desktop\apex\Internal
```

then type

```shell
kdmapper driver.sys
```

then type `modmap "process name".exe "legit dll already injected".dll "name of your dll".dll`。For example:

```shell
modmap r5apex.exe dxgi.dll ApexInternal.dll
# or
modmap r5apex.exe Activation64.dll ApexInternal.dll
```

Here is the download: [https://www.unknowncheats.me/forum/d...=file&id=29360](https://www.unknowncheats.me/forum/downloads.php?do=file&id=29360)

I got the list of loaded dlls in apex with process hacker. If the dll is **too small** you will get this error "module does not having enough free trailing memory (C0000141)".  `module dxgi.dll does not having enough free trailing memory (C0000141)`。You can't inject **big** dlls wich are 3mb for example.

```c++
XAudio2_6.dll
wshbth.dll
ws2_32.dll
Windows.UI.dll
Windows.Internal.Graphics.Display.DisplayColorManagement.dll
imagehlp.dll
msvcrt.dll
dbghelp.dll
bcryptprimitives.dll
dbgcore.dll
win32u.dll
version.dll
iertutil.dll
userenv.dll
profapi.dll
ksuser.dll
normaliz.dll
propsys.dll
steam_api64.dll
SHCore.dll
AudioSes.dll
winhttp.dll
dhcpcsvc.dll
sspicli.dll
secur32.dll
rpcrt4.dll
ncrypt.dll
ResourcePolicyClient.dll
rasadhlp.dll
bink2w64.dll
wdmaud.drv
wdmaud.drv.mui
igo64.dll
oleaut32.dll
nvapi64.dll
MessageBus.dll
nvspcap64.dll
nvldumdx.dll
nvwgf2umx.dll
nvcuda.dll
nsi.dll
winnsi.dll
nlaapi.dll
imm32.dll
binkawin64.dll
mileswin64.dll
msvcp_win.dll
ucrtbase.dll
vcruntime140.dll
msvcp140.dll
vcruntime140_1.dll
WindowsCodecs.dll
wintrust.dll
ncryptsslp.dll
ole32.dll
midimap.dll
GdiPlus.dll
rsaenh.dll
dinput8.dll
dinput8.dll
CoreMessaging.dll
CoreUIComponents.dll
xinput1_3.dll
xinput1_3.dll
combase.dll
coloradapterclient.dll
ntasn1.dll
msacm32.drv
winrnr.dll
MpOAV.dll
setupapi.dll
InputHost.dll
sechost.dll
gdi32.dll
gdi32full.dll
NapiNSP.dll
ntmarta.dll
pnrpnsp.dll
schannel.dll
mswsock.dll
mswsock.dll.mui
mskeyprotect.dll
msacm32.dll
avrt.dll
urlmon.dll
wininet.dll
EasyAntiCheat_x64.dll
Activation64.dll
DXCore.dll
dnsapi.dll
msvfw32.dll
msvfw32.dll.mui
powrprof.dll
mscms.dll
kernel32.dll
KernelBase.dll
KernelBase.dll
KernelBase.dll.mui
kernel32.dll.mui
WinTypes.dll
winnlsres.dll.mui
winnlsres.dll
msctf.dll
ntdll.dll
ntdll.dll
shell32.dll
shell32.dll
user32.dll
user32.dll
winmm.dll
Wldap32.dll
dxgi.dll
dxgi.dll
dsound.dll
d3dcompiler_47_64.dll
D3DCompiler_43.dll
d3d11.dll
devobj.dll
dpapi.dll
cryptsp.dll
cryptnet.dll
crypt32.dll
crypt32.dll.mui
XInput9_1_0.dll
cfgmgr32.dll
clbcatq.dll
NvCamera64.dll
shlwapi.dll
hid.dll
avifil32.dll
bcrypt.dll
comctl32.dll
winmmbase.dll
cryptbase.dll
msasn1.dll
kernel.appcore.dll
MMDevAPI.dll
MMDevAPI.dll.mui
FWPUCLNT.DLL
dwmapi.dll
webio.dll
windows.storage.dll
IPHLPAPI.DLL
gpapi.dll
advapi32.dll
advapi32.dll
r5apex.exe
amsi.dll
TextInputFramework.dll
umpdc.dll
```



评论



`load driver => inject => unload driver`，**unloaded drivers would show in mmunloadeddrivers and such so you'd have to clean those out aswell**



`secur32.dll` 非常适合我，非常感谢！



The injector still work but the internal p200 need an update so it crash





