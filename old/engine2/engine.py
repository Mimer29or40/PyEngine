import time
from . import events


class Game:
    def __init__(self):
        self.name = self.__class__.__name__
    
    def init(self):
        pass
    
    def process_events(self, events):
        for event in events:
            if event.type == events.QUIT:
                stop()
    
    def update(self, t, dt):
        pass
    
    def render(self, t, dt):
        pass
    
    def shutdown(self):
        pass


def set_game(game):
    global _game
    if _running:
        raise Exception('Engine is Running')
    _game = game


def get_tps():
    return _tps_actual


def get_fps():
    return _fps_actual


def set_tps(tps):
    global _tps_target, _tps_actual, _spt_target
    _tps_target = tps
    _tps_actual = tps
    _spt_target = 0 if tps is None or tps < 1 else int(1e9 / tps)


def set_fps(fps):
    global _fps_target, _fps_actual, _spf_target
    _fps_target = fps
    _fps_actual = fps
    _spf_target = 0 if fps is None or fps < 1 else int(1e9 / fps)


def get_start_time():
    return _start_time


def get_time():
    return 0 if _start_time < 0 else time.perf_counter_ns() - _start_time


def run():
    global _start_time, _running, _tps_actual, _fps_actual
    try:
        _game.init()
        
        _start_time = time.perf_counter_ns()
        _running = True
        
        tick_count = 0
        frame_count = 0
        
        last_tick = get_time()
        last_frame = get_time()
        last_sec = get_time()
        
        while _running:
            _game.process_events(events.get_events())
            
            t = get_time()
            dt = t - last_tick
            if dt >= _spt_target:
                tick_count += 1
                last_tick = t
                
                _game.update(t / 1e9, dt / 1e9)
            
            t = get_time()
            dt = t - last_frame
            if dt >= _spf_target:
                frame_count += 1
                last_frame = t
                
                _game.render(t / 1e9, dt / 1e9)
            
            t = get_time()
            dt = t - last_sec
            if dt >= 1e9:
                last_sec = t
                
                _tps_actual = tick_count
                _fps_actual = frame_count
                
                tick_count = 0
                frame_count = 0
        
    except Exception as e:
        # print(e)
        raise
    
    _game.shutdown()


def stop():
    global _running
    _running = False


_game = Game()

_tps_target, _tps_actual, _spt_target = None, 0, 0
_fps_target, _fps_actual, _spf_target = None, 0, 0

set_tps(60)
set_fps(60)

_start_time = -1
_running = False
