import cv2
import pickle,os
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.measure import regionprops,label
from matplotlib.patches import Rectangle
from train import cal_probG
import pandas as pd
from sklearn import linear_model

testImFl = 'Test_Set'
testClass=['RedBarrel','RedWall','RedChairs','OrangeFloor','RedObstacle','RedCokeMach','RedMachine', 'RedSofa']
numClass =len(testClass)


def calPrior(pX_Cl):
    pc=[0.25, 0.0005,0.001, 0.1,0.01,0.004,0.005,0.08]
    # px = np.zeros_like(pX_Cl)
    return pc

def loadTest(testImList, ii):

    test_im = cv2.imread(testImFl+'/'+testImList[ii])
    test_im = cv2.cvtColor(test_im,cv2.COLOR_BGR2HLS)
    return test_im


def pred(test_im,mu,A,alpha,n_subclass):
    H,W,_=np.shape(test_im)
    test_data = np.zeros([H*W,2])
    test_data[:,[0,1]]=test_im[:,:,[0,2]].reshape(-1,2)
    rr= np.zeros([H*W,n_subclass])

    for k in range(n_subclass):
        rr[:, k] = alpha[k] * cal_probG(test_data,A[k],mu[k])

    rr = rr.sum(axis=1).reshape(H,W)

    return rr


def train_regressor(barrel_dict):
    df = pd.DataFrame(data=barrel_dict)
    y_train=df['dis'].reshape(-1,1)
    df['1/area']=1/df['area']
    x_train=df['1/area'].reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    y_pre=regr.predict(x_train)
    df['dis_pre']=y_pre
    pickle.dump(regr,open('regressor.pkl','wb'))

    return y_pre

def test_regressor(area):

    x_train = (1/np.array([area])).reshape(-1, 1)
    regr = pickle.load(open('regressor.pkl', 'rb'))
    y_pre = regr.predict(x_train)

    return y_pre[0,0]


if __name__=='__main__':
    testImList = [file for file in os.listdir(testImFl) if file.endswith('.png')]


    barrel_dict={}
    barrel_dict['Im'] = []
    barrel_dict['centerX'] = []
    barrel_dict['centerY'] = []
    barrel_dict['dis'] = []
    # barrel_dict['area'] = []
    for i in range(len(testImList)):
        test_im = loadTest(testImList, i)
        H, W, _ = np.shape(test_im)
        re = np.zeros([H, W, numClass])
        mu,A,alpha,lg,n_subclass=[],[],[],[],[]
        for m in range(numClass):
            para = pickle.load(open(testClass[m] + '.pkl', 'rb'))
            mu.append(para['mu'])
            A.append(para['A'])
            alpha.append(para['alpha'])
            lg.append(para['lg'])
            n_subclass.append(para['n_subclass'])

            re[:,:,m] = pred(test_im, mu[m], A[m], alpha[m],n_subclass[m])

        pc=calPrior(re)
        for n in range(numClass):
            re[:,:,n]*=pc[n]


        max_coor=np.argmax(re, axis=2)
        barrel=np.zeros([H,W])
        barrel[np.where(max_coor==0)]=1
        barrel=binary_erosion(barrel,structure=np.ones((10,10)))
        barrel = binary_dilation(barrel, structure=np.ones((30, 30)))
        barrel = binary_erosion(barrel, structure=np.ones((10, 10)))

        # plt.imshow(barrel)
        # plt.show()

        label_im=label(np.array(barrel, dtype=int))
        regions = regionprops(label_im)

        area=np.empty([0])
        valid=[]
        for h, pop in enumerate(regions):
            area=np.append(area,pop.area)
        if h==0:
            valid.append(0)
        elif h>0:
            ind_d=np.argsort(area)
            area_d =np.sort(area)
            valid.append(ind_d[-1])
            ratio_area=area_d[:-1]/area_d[1:]
            for kk in range(len(ratio_area)):
                if ratio_area[len(ratio_area)-kk-1]>0.4:
                    valid.append(ind_d[len(ratio_area)-kk-1])
                else:
                    break

        fig, ax = plt.subplots(1)
        atleast_sign = 0
        new_valid=valid.copy()
        for f in range(len(valid)):

            minr, minc, maxr, maxc = regions[valid[f]].bbox
            if 1.1<(maxr-minr)/(maxc-minc)<2.1:
                atleast_sign=1
                ax.imshow(cv2.cvtColor(test_im,cv2.COLOR_HLS2RGB))
                rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red',linewidth=2)
                ax.add_patch(rect)
            else:del new_valid[f]
        if atleast_sign==0 and len(ind_d)>1:
            new_valid.append(ind_d[-2])
            minr, minc, maxr, maxc = regions[ind_d[-2]].bbox
            ax.imshow(cv2.cvtColor(test_im, cv2.COLOR_HLS2RGB))
            rect = Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        # plt.show()


        num_barrel = len(new_valid)
        seg = np.zeros([H, W])
        for mm in range(num_barrel):
            barrel_dict['Im'].append(testImList[i])

            minr, minc, maxr, maxc = regions[new_valid[mm]].bbox
            mj_len,mn_len=regions[new_valid[mm]].major_axis_length,regions[new_valid[mm]].minor_axis_length
            area_sp=((maxr - minr) * (maxc - minc) + regions[new_valid[mm]].area + mj_len * mn_len * 4) / 3.
            # barrel_dict['area'].append(area_sp)
            barrel_dict['dis'].append(test_regressor(area_sp))
            # minr, minc, maxr, maxc = regions[new_valid[mm]].bbox
            r, c = regions[new_valid[mm]].local_centroid
            ax.plot(c+minc,r+minr,'.')

            barrel_dict['centerX'].append(c+minc)
            barrel_dict['centerY'].append(r+minr)
            # seg=np.zeros([H,W])
            a = regions[new_valid[mm]].coords[:, 0]
            b = regions[new_valid[mm]].coords[:, 1]
            seg[a,b]=1
        # plt.show()
        plt.savefig('bbox'+testImList[i])
        plt.imshow(seg)
        plt.savefig('seg' + testImList[i])
        # plt.show()
        df=pd.DataFrame(data=barrel_dict)
        df.to_csv('DataFrame_for_testingset.csv')
        print(df)


    #
    # #Used to train regressor
    #     for mm in range(num_barrel):
    #         barrel_dict['Im'].append(testImList[i])
    #         if num_barrel==1:
    #             barrel_dict['dis'].append(float(testImList[i].split('.',1)[0]))
    #         else:
    #             dis=testImList[i].split('_',num_barrel)
    #             if mm==num_barrel-1:barrel_dict['dis'].append(float(dis[-1].split('.',1)[0]))
    #             else:barrel_dict['dis'].append(float(dis[mm]))
    #
    #         minr, minc, maxr, maxc=regions[new_valid[mm]].bbox
    #         mj_len,mn_len=regions[new_valid[mm]].major_axis_length,regions[new_valid[mm]].minor_axis_length
    #         barrel_dict['area'].append(((maxr-minr)*(maxc-minc)+regions[new_valid[mm]].area+mj_len*mn_len*4)/3.)
    #         r,c=regions[new_valid[mm]].local_centroid
    #         barrel_dict['centerX'].append(c)
    #         barrel_dict['centerY'].append(r)
    #
    #
    # train_regressor(barrel_dict)
    #
    # print(barrel_dict)


