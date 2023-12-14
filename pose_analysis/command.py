import pygetwindow as gw
def command(action):
    if (action ==0):
        send_key_to_window("Citra Nightly 2052 | Super Smash Bros.", "a")
    elif (action ==1):
        send_key_to_window("Citra Nightly 2052 | Super Smash Bros.", "n")
    elif (action ==2):
        i = None

def send_key_to_window(window_title, key):
    # Get the window by title
    target_window = gw.getWindowsWithTitle(window_title)
    
    if not target_window:
        print(f"Window with title '{window_title}' not found.")
        return
    
    target_window = target_window[0]
    target_window.activate()


def command(action):
    if (action ==0):
        send_key_to_window("Citra Nightly 2052 | Super Smash Bros.", "a")
    elif (action ==1):
        send_key_to_window("Citra Nightly 2052 | Super Smash Bros.", "n")
    elif (action ==2):
        i = None

def send_key_to_window(window_title, key):
    # Get the window by title
    target_window = gw.getWindowsWithTitle(window_title)
    
    if not target_window:
        print(f"Window with title '{window_title}' not found.")
        return
    
    target_window = target_window[0]
    target_window.activate()



    # Send the key press
    pyautogui.keyDown(key)
    pyautogui.keyUp(key)
