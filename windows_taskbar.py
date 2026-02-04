import ctypes
from ctypes import wintypes
import sys
from enum import IntEnum

# Windows GUIDs and Constants
CLSID_TaskbarList = ctypes.POINTER(ctypes.c_int)

# ITaskbarList3 Constants
class TBPFLAG(IntEnum):
    TBPF_NOPROGRESS = 0
    TBPF_INDETERMINATE = 1
    TBPF_NORMAL = 2
    TBPF_ERROR = 4
    TBPF_PAUSED = 8

CLSID_TaskbarList = ctypes.c_ubyte * 16
IID_ITaskbarList3 = ctypes.c_ubyte * 16

# {56FDF344-FD6D-11d0-958A-006097C9A090}
CLSID_TaskbarList_UUID = bytes(
    [0x44, 0xF3, 0xFD, 0x56, 0x6D, 0xFD, 0xd0, 0x11, 0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90]
)

# {ea1afb91-9e28-4b86-90e9-9e9f8a5eefaf}
IID_ITaskbarList3_UUID = bytes(
    [0x91, 0xFB, 0x1A, 0xEA, 0x28, 0x9E, 0x86, 0x4B, 0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF]
)

class TaskbarController:
    def __init__(self):
        self._tbl = None
        self._initialized = False
        
        if sys.platform != "win32":
            return

        try:
            # Initialize COM
            ctypes.windll.ole32.CoInitialize(0)
            
            # Create the TaskbarList object
            self._tbl = ctypes.POINTER(ctypes.c_void_p)()
            
            clsid = CLSID_TaskbarList.from_buffer_copy(CLSID_TaskbarList_UUID)
            iid = IID_ITaskbarList3.from_buffer_copy(IID_ITaskbarList3_UUID)
            
            ctypes.windll.ole32.CoCreateInstance(
                ctypes.byref(clsid),
                0,
                1,  # CLSCTX_INPROC_SERVER
                ctypes.byref(iid),
                ctypes.byref(self._tbl)
            )
            
            # Helper to get the vtable function
            def get_vtbl_func(interface_ptr, index, argtypes, restype):
                vtbl_ptr = ctypes.cast(interface_ptr.contents, ctypes.POINTER(ctypes.c_void_p))
                func_ptr = vtbl_ptr[index]
                func = ctypes.CFUNCTYPE(restype, *argtypes)(func_ptr)
                return func

            # ITaskbarList3 methods
            # 3: HrInit
            # 9: SetProgressValue
            # 10: SetProgressState
            
            self._hr_init = get_vtbl_func(
                self._tbl, 3, [ctypes.c_void_p], ctypes.c_int
            )
            
            self._set_progress_value = get_vtbl_func(
                self._tbl, 9, 
                [ctypes.c_void_p, wintypes.HWND, ctypes.c_ulonglong, ctypes.c_ulonglong], 
                ctypes.c_int
            )
            
            self._set_progress_state = get_vtbl_func(
                self._tbl, 10, 
                [ctypes.c_void_p, wintypes.HWND, ctypes.c_int], 
                ctypes.c_int
            )

            # Initialize
            self._hr_init(self._tbl)
            self._initialized = True
            
        except Exception as e:
            print(f"Failed to initialize TaskbarController: {e}")
            self._tbl = None

    def set_app_id(self, app_id: str):
        if sys.platform == "win32":
            try:
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            except Exception:
                pass

    def set_progress_state(self, hwnd: int, state: TBPFLAG):
        if self._initialized and self._tbl:
            try:
                self._set_progress_state(self._tbl, hwnd, state)
            except Exception:
                pass

    def set_progress_value(self, hwnd: int, completed: int, total: int):
        if self._initialized and self._tbl:
            try:
                self._set_progress_value(self._tbl, hwnd, completed, total)
            except Exception:
                pass

    def get_hwnd(self, root_tk) -> int:
        """Helper to get HWND from a Tk instance"""
        try:
            return int(root_tk.frame(), 16)
        except Exception:
            # Fallback for some Tk versions or if frame() returns id
            try:
                return root_tk.winfo_id()
            except Exception:
                return 0
