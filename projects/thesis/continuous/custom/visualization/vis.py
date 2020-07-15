import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math

__all__ = [
    "visualize_weight", "visualize_feature", "register_feature"
]

def visualize_weight(model, idx, path=None,  cols=None, show=False):

    """Visualize weight and activation matrices learned
    during the optimization process. Works for any size of kernels.

    Arguments
    =========
    kernels: Weight or activation matrix. Must be a high dimensional
    Numpy array. Tensors will not work.
    path: Path to save the visualizations.
    cols: TODO: Number of columns (doesn't work completely yet.)

    """
    model_state = model.state_dict()
    if not os.path.exists(path):
        os.makedirs(path)

    for name, kernels in model_state.items():

        if ("conv" in name) and ("weight" in name):
            fpath = path + "e" + str(idx + 1) + "/" + str(name) + "/"
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            kernels = kernels.cpu().detach().clone().numpy()
            kernels = kernels - kernels.min()
            kernels = kernels / kernels.max()

            N = kernels.shape[0]
            C = kernels.shape[1]
            Tot = N * C  # inchannel, outchannel

            cols = C
            rows = Tot // cols + 1
            pos = range(1, Tot + 1)

            for t in range(kernels.shape[2]):
                k = 0
                fig = plt.figure(t)
                fig.tight_layout()
                for i in range(kernels.shape[0]):
                    for j in range(kernels.shape[1]):
                        img = kernels[i][j][t]
                        ax = fig.add_subplot(rows, cols, pos[k])
                        ax.imshow(img, cmap='gray')
                        plt.axis('off')
                        k = k + 1
                # if not os.path.exists(path + "t"+ str(m)):
                #     os.makedirs(path + "t" + str(m))
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                if fpath:
                    plt.savefig(fpath + "t" + str(t) + ".png", dpi=100)
                    plt.close(fig)
                if show:
                    plt.show()




def register_feature(features):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for key, feature in features.named_modules():
        feature.register_forward_hook(get_activation(key))

    return activation

def visualize_feature(input_data, features, path, idx, show=False):
    if not os.path.exists(path):
        os.makedirs(path)

    for name, feature in features.items():
        if name in ['fc1', 'fc2', 'fc3', 'dropout', '', ]:
            continue
        fpath = path + "e" + str(idx+1) + "/" + str(name) + "/"
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        feature = feature.cpu().detach().clone().numpy()
        feature = feature[0]
        feature = feature - feature.min()
        feature = feature / feature.max()
        C, T, H, W = feature.shape

        cols = int(math.sqrt(C))
        rows = C // cols + 1
        pos = range(1, C + 1)

        for m in range(T):
            k = 0
            fig = plt.figure(m)
            fig.tight_layout()
            for i in range(C):
                img = feature[i][m]
                ax = fig.add_subplot(rows, cols, pos[k])
                # ax.plot(img)
                ax.imshow(img, cmap='gray')
                plt.axis('off')
                k = k + 1

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            if fpath:
                plt.savefig(fpath + "/t" + str(m) + ".png", dpi=100)
                plt.close(fig)
            if show:
                plt.show()

    fpath = path + "e" + str(idx+1) + "/" + "input_data" + "/"
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    input_data = input_data.unsqueeze(1)
    input_data = input_data.cpu().detach().clone().numpy()
    input_data = input_data[0]
    input_data = input_data - input_data.min()
    input_data = input_data / input_data.max()
    C, T, H, W = input_data.shape

    cols = int(math.sqrt(C))
    rows = C // cols + 1
    pos = range(1, C + 1)

    for m in range(T):
        k = 0
        fig = plt.figure(m)
        fig.tight_layout()
        for i in range(C):
            img = input_data[i][m]
            ax = fig.add_subplot(rows, cols, pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k + 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if fpath:
            plt.savefig(fpath + "/t" + str(m) + ".png", dpi=100)
            plt.close(fig)
        if show:
            plt.show()


# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# img = [] # some array of images
# frames = [] # for storing the generated images
# fig = plt.figure()
# for i in xrange(6):
#     frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])
#
# ani = animation.ArtistAnimation(fig, img, interval=50, blit=True,
#                                 repeat_delay=1000)
# # ani.save('movie.mp4')
# plt.show()


# def generate_video(img):
#     for i in xrange(len(img)):
#         plt.imshow(img[i], cmap=cm.Greys_r)
#         plt.savefig(folder + "/file%02d.png" % i)
#
#     os.chdir("your_folder")
#     subprocess.call([
#         'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
#         'video_name.mp4'
#     ])
#     for file_name in glob.glob("*.png"):
#         os.remove(file_name)