from pynput import keyboard
from pynput.mouse import Controller, Button
from functools import partial
from src.settings import STACK_MODIFIER_KEYS, STACK_TOWER_KEYS, STACK_MINION_KEY, QUIT_MODIFIER_KEYS, QUIT_KEY, HOLD_MOUSE_KEY
from src.consts import VALID_MODIFIER_KEYS

mouse = Controller()
kb = keyboard.Controller()


def stack_minion():
    mouse.click(Button.left, 2)
    

def hold_mouse():
    mouse.press(Button.right)


def on_key_pressed(key):
    kb.type(f"{key} ")


def setup_hotkeys():
    stack_modifier_set = set(STACK_MODIFIER_KEYS)
    quit_modifier_set = set(QUIT_MODIFIER_KEYS)
    for key in stack_modifier_set:
        if key not in VALID_MODIFIER_KEYS:
            raise Exception(f"Invalid modifier key in STACK_MODIFIER_KEYS: '{key}'")
    for key in quit_modifier_set:
        if key not in VALID_MODIFIER_KEYS:
            raise Exception(f"Invalid modifier key in QUIT_MODIFIER_KEYS: '{key}'")
    hotkeys = {"".join([f"<{mod_key}>+" for mod_key in stack_modifier_set]) + key: partial(on_key_pressed, key) for key
               in STACK_TOWER_KEYS}
    hotkeys["".join([f"<{mod_key}>+" for mod_key in stack_modifier_set]) + STACK_MINION_KEY] = stack_minion
    hotkeys["".join([f"<{mod_key}>+" for mod_key in stack_modifier_set]) + HOLD_MOUSE_KEY] = hold_mouse
    hotkeys["".join([f"<{mod_key}>+" for mod_key in quit_modifier_set]) + QUIT_KEY] = exit
    print(
        f"Hold '{'+'.join(stack_modifier_set)}' and press {STACK_TOWER_KEYS} to stack a tower " +
        f"or '{STACK_MINION_KEY}' to stack a minion.\n" +
        f"Hold '{'+'.join(stack_modifier_set)}' and press '{HOLD_MOUSE_KEY}' to enable holding down the right mouse button.\n" +
        f"Hold '{'+'.join(quit_modifier_set)}' and press '{QUIT_KEY}' to quit.")
    with keyboard.GlobalHotKeys(hotkeys) as h:
        h.join()
