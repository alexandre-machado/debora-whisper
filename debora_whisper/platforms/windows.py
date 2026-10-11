"""Windows provider.

Each playback endpoint's form factor is in the registry, so telling
headphones from speakers needs no extra package.
"""
import subprocess

from debora_whisper.platforms.base import Platform

_RENDER = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
_DESCRIPTION = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"  # "Headphones"
_DEVICE_NAMES = ("{b3f8fa53-0004-438e-9003-51a46e139bfc},26",  # "LE-HD 450BT"
                 "{b3f8fa53-0004-438e-9003-51a46e139bfc},6")   # "Realtek(R) Audio"
_FORM_FACTOR = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},0"
_ACTIVE = 1
_HEADPHONES, _HEADSET = 3, 5
# MME, sounddevice's default host API, cuts device names at 31 characters.
_MME_NAME_CHARS = 31


def render_endpoints():
    """[(name as audio APIs show it, form factor)] of active playback devices."""
    import winreg
    endpoints = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _RENDER)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                guid = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, guid) as key:
                    if winreg.QueryValueEx(key, "DeviceState")[0] != _ACTIVE:
                        continue
                with winreg.OpenKey(root, guid + r"\Properties") as props:
                    values = {}
                    for name in (_DESCRIPTION, _FORM_FACTOR, *_DEVICE_NAMES):
                        try:
                            values[name] = winreg.QueryValueEx(props, name)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            device = next((values[n] for n in _DEVICE_NAMES if values.get(n)), None)
            if values.get(_DESCRIPTION) and device:
                endpoints.append((f"{values[_DESCRIPTION]} ({device})",
                                  values.get(_FORM_FACTOR)))
    return endpoints


def is_headphones(name, endpoints):
    """True/False by the matching endpoint's form factor, None if not found."""
    if not name:
        return None
    matches = {factor for full, factor in endpoints
               if full == name or (len(name) >= _MME_NAME_CHARS and full.startswith(name))}
    if len(matches) != 1:
        return None
    return matches.pop() in (_HEADPHONES, _HEADSET)


class WindowsPlatform(Platform):
    name = "windows"

    @property
    def no_window(self):
        return subprocess.CREATE_NO_WINDOW

    def kill_tree(self, process):
        """Kill process and its children, without waiting for them.

        On Windows a venv's python.exe and uv both run the real interpreter as a
        child: killing only the parent leaves that child running, holding its
        GPU or NPU."""
        if process.poll() is not None:
            return
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True, creationflags=self.no_window)

    def output_is_headphones(self, name=None):
        name = self.default_output_name() if name is None else name
        return is_headphones(name, render_endpoints())
