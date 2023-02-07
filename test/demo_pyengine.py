from pyengine import *


@instance
class DemoEngine(AbstractEngine):
    def setup(self) -> None:
        pass

    def update(self, time: float, delta_time: float) -> None:
        if window_on_close():
            print("window_on_close()")
        if window_on_focused():
            print(f"window_on_focused()={window_focused()}")
        if window_on_minimized():
            print(f"window_on_minimized()={window_minimized()}")
        if window_on_maximized():
            print(f"window_on_maximized()={window_maximized()}")
        if window_on_pos_change():
            print("window_on_pos_change()=" + window_pos())
        if window_on_size_change():
            print("window_on_size_change()=" + window_size())
        # if window_on_content_scale_change():
        #     print("window_on_content_scale_change()=" + window_content_scale())
        # if window_on_framebuffer_size_change():
        #     print("window_on_framebuffer_size_change()=" + window_framebuffer_size())
        # if window_on_refresh():
        #     print("window_on_refresh()")
        # if window_on_dropped():
        #     print("window_on_dropped()=" + window_dropped())

        # if mouse_on_entered():
        #     print("mouse_on_entered()" + mouse_entered())
        # if mouse_on_pos_change():
        #     print("mouse_on_pos_change()=" + mouse_pos() + mouse_pos_delta())
        # if mouse_on_scroll_change():
        #     print("mouse_on_scroll_change()=" + mouse_scroll())
        # if mouse_on_button_down():
        #     print("mouse_on_button_down()=" + mouse_button_down())
        # if mouse_on_button_up():
        #     print("mouse_on_button_up()=" + mouse_button_up())
        # if mouse_on_button_repeated():
        #     print("mouse_on_button_repeated()=" + mouse_button_repeated())
        # if mouse_on_button_held():
        #     print("mouse_on_button_held()=" + mouse_button_held())
        # if mouse_on_button_dragged():
        #     print("mouse_on_button_dragged()=" + mouse_button_dragged())

        # if keyboard_on_typed():
        #     print("keyboard_on_typed()=" + keyboard_typed())
        # if keyboard_on_key_down():
        #     print("keyboard_on_key_down()=" + keyboard_key_down())
        # if keyboard_on_key_up():
        #     print("keyboard_on_key_up()=" + keyboard_key_up())
        # if keyboard_on_key_repeated():
        #     print("keyboard_on_key_repeated()=" + keyboard_key_repeated())
        # if keyboard_on_key_held():
        #     print("keyboard_on_key_held()=" + keyboard_key_held())

        # if keyboard_key_down(Key.SPACE) and modifier_only(Modifier.SHIFT):
        #     if mouse_is_shown():
        #         mouse_hide()
        #     elif mouse_is_hidden():
        #         mouse_capture()
        #     elif mouse_is_captured():
        #         mouse_show()

        # if mouse_button_down(Button.LEFT):
        #     print("LEFT " + mouse_button_down_count(Button.ONE))
        # if mouse_button_up(Button.MIDDLE):
        #     print("MIDDLE Up")
        # if mouse_button_repeated(Button.RIGHT):
        #     print("RIGHT Repeated")
        # if mouse_button_held(Button.FOUR):
        #     print("FOUR Held")
        # if mouse_button_dragged(Button.FIVE):
        #     print("FIVE Held")

        # if keyboard_key_down(Key.W):
        #     print("W Down " + keyboard_key_down_count(Key.W))
        # if keyboard_key_up(Key.A):
        #     print("A Up")
        # if keyboard_key_repeated(Key.S):
        #     print("S Repeated")
        # if keyboard_key_held(Key.D):
        #     print("D Held")

    def draw(self, time: float, delta_time: float) -> None:
        # stop()
        pass

    def destroy(self) -> None:
        pass


draw_rate(60)

print(viewport())

start()
