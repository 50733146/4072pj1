import gym
import pygame
import matplotlib
import argparse
from gym import logger #?
#try-except 陳述 (try-except statement):凡是可能會產生例外的程式碼
#try:可能會產生例外的程式碼
#expect:例外發生時的處置
try:
    matplotlib.use('TkAgg') #?
    import matplotlib.pyplot as plt
#%s使用str()將字串輸出
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None
#collections:容器資料型態
#一個類似 list 的容器，可以快速的在頭尾加入元素與取出元素
from collections import deque
#from pygame.locals import VIDEORESIZE #!


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    #pygame.surfarray.make_surface:Copy an array to a new surface
    #.swapaxes:互換數組的兩個軸。
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    #pygame.transform.scale:resize to new resolution(解析度)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    #.blit(背景變數, 繪製位置)
    screen.blit(pyg_img, (0, 0))


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.

    To simply play the game use:

        play(gym.make("Pong-v4"))

    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.

        def callback(obs_t, obs_tp1, action, rew, done, info):
            return [rew,]
        plotter = PlayPlot(callback, 30 * 5, ["reward"])

        env = gym.make("Pong-v4")
        play(env, callback=plotter.callback)


    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:

            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    env.reset()
    rendered = env.render(mode='rgb_array')

    if keys_to_action is None:
        #hasattr()用於判斷對象是否包含的屬性;對象有該屬性返回True,否則返回False
        if hasattr(env, 'get_keys_to_action'):
        #指定自定義鍵以進行動作映射
            keys_to_action = env.get_keys_to_action()
        #env.unwrapped:環境變量
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
        #keys_to_action等於環境中可用的action
            keys_to_action = env.unwrapped.get_keys_to_action()
        #assert:用來進行簡單的條件 (condition) 判斷，如果條件為真，程式 (program) 繼續執行，
        #反之條件為假的話，就會發起 Assertion 例外 (exception) ，中斷程式執行。
        #逗號後面則是 Assertion 例外的提示訊息，假如是Ture就不會觸發例外
        #env.spec.id:拆開環境並從規格中獲取ID
        #字符串聯接，所有的字符串都是直接按照字面的意思来使用，没有轉為特殊或不能打印的字符。
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
            #set:建立集合，集合不會包含重複的資料
            #map:會根據提供的函數，對指定序列做映射
            #list:儲存一連串有順序性的元素
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
   #.shape:功能是查看矩阵或者数组的维数
    video_size = [rendered.shape[1], rendered.shape[0]]
#int()(interger):將傳入之參數轉為整數，若參數為浮點數則將小數捨去
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            #tuple:功能與list相同，但Tuple資料組建立之後，裏頭的資料就不可以再改變
            #Tuple資料組中的資料可以有不同的資料型態，甚至可以包含另一個資料組。
            #Tuple資料組可以利用指定運算子把資料分配給多個變數
            #Tuple資料組顯示的時候，會用圓括弧把資料括起來
            #sorted:對所有可迭代的對象進行排序(小到大)操作
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            rendered = env.render(mode='rgb_array')
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:

                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False

            '''elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)'''


        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()


class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        assert plt is not None, "matplotlib backend failed, plotting will not work"
#len返回对象（字符、列表、元组等）长度或项目个数
        num_plots = len(self.plot_names)
        # 用來輸出 總畫布“視窗”的（fig：是figure的縮寫），有了畫布就可以在上邊作圖。
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        #zip與for迴圈:將兩個 list 以迴圈的方式一次各取一個元素出來處理
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)#將標題新增到 Matplotlib 中的各個子圖中
        self.t = 0
        #None:表示不存在
        self.cur_plot = [None for _ in range(num_plots)]
        #’_’ 是一个循环标志，也可以用i，j 等其他字母代替
        #deque:指定一個maxlen參數，限制deque的大小，但並不是只要deque滿了就不行再接收新的元素，而是從哪一端獲取新元素就從另一端把舊元素丟棄
        #返回的是一个可迭代对象（類型是对象），而不是列表類型， 所以打印的时候不会打印列表。
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1
#max:返回给定参数的最大值
        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t
#enumerate:将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                #.remove():從 Matplotlib 中的圖形中刪除圖例
                plot.remove()
                #plt.scatter(x軸數列, y軸數列, alpha=透明度, label = ‘散點名稱’)畫散點圖
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]), c='blue')
            self.ax[i].set_xlim(xmin, xmax)
            #plt.pause:暫停間格秒
        plt.pause(0.000001)


def main():
    # ArgumentParser:对象包含将命令行解析成 Python 数据类型所需的全部信息
    parser = argparse.ArgumentParser()
# add_argument():利用這個方法可以指名讓我們的程式接受哪些命令列參數
    parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4', help='Define Environment')
    args = parser.parse_args()#返回具有env屬性的對象
    env = gym.make(args.env)
    play(env, zoom=4, fps=60)

#可能會有「單獨執行」與「被引用」兩種情形
if __name__ == '__main__':#此判斷式，即可讓檔案在被引用時，不該執行的程式碼會不被執行
    main()