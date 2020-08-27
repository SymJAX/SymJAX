.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_compare_adam.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_compare_adam.py:


Adam TF and SymJAX
==================

In this example we demonstrate how to perform a simple optimization with Adam in TF and SymJAX



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_001.svg
          :alt: plot compare adam
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_002.svg
          :alt: Adam Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_003.svg
          :alt: GD Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=true_divide, fn=true_divide, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/), Placeholder(name=x, shape=(), dtype=float32, scope=/), Variable(name=num_steps, shape=(), dtype=int32, trainable=False, scope=/ExponentialMovingAverage/)]
    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=add, fn=add, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Placeholder(name=x, shape=(), dtype=float32, scope=/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/)]
      0%|          | 0/400 [00:00<?, ?it/s]     18%|#8        | 74/400 [00:00<00:00, 738.41it/s]     42%|####1     | 167/400 [00:00<00:00, 786.52it/s]     67%|######7   | 268/400 [00:00<00:00, 841.21it/s]     90%|######### | 362/400 [00:00<00:00, 868.31it/s]    100%|##########| 400/400 [00:00<00:00, 897.45it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<02:14,  2.96it/s]     10%|#         | 41/400 [00:00<01:25,  4.22it/s]     21%|##1       | 85/400 [00:00<00:52,  6.00it/s]     32%|###2      | 130/400 [00:00<00:31,  8.52it/s]     44%|####3     | 174/400 [00:00<00:18, 12.07it/s]     55%|#####4    | 218/400 [00:00<00:10, 17.04it/s]     66%|######6   | 266/400 [00:00<00:05, 23.97it/s]     78%|#######8  | 314/400 [00:01<00:02, 33.53it/s]     91%|######### | 363/400 [00:01<00:00, 46.51it/s]    100%|##########| 400/400 [00:01<00:00, 326.94it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     18%|#8        | 74/400 [00:00<00:00, 734.15it/s]     44%|####4     | 176/400 [00:00<00:00, 800.57it/s]     70%|######9   | 278/400 [00:00<00:00, 854.12it/s]     95%|#########4| 379/400 [00:00<00:00, 895.55it/s]    100%|##########| 400/400 [00:00<00:00, 944.07it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<02:10,  3.06it/s]     13%|#3        | 52/400 [00:00<01:19,  4.37it/s]     24%|##4       | 96/400 [00:00<00:48,  6.21it/s]     36%|###5      | 142/400 [00:00<00:29,  8.82it/s]     48%|####7     | 190/400 [00:00<00:16, 12.50it/s]     60%|######    | 240/400 [00:00<00:09, 17.67it/s]     72%|#######2  | 289/400 [00:00<00:04, 24.85it/s]     84%|########3 | 335/400 [00:01<00:01, 34.70it/s]     96%|#########5| 382/400 [00:01<00:00, 48.04it/s]    100%|##########| 400/400 [00:01<00:00, 342.04it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     18%|#8        | 74/400 [00:00<00:00, 733.99it/s]     44%|####3     | 174/400 [00:00<00:00, 796.45it/s]     69%|######8   | 275/400 [00:00<00:00, 849.44it/s]     94%|#########3| 376/400 [00:00<00:00, 890.69it/s]    100%|##########| 400/400 [00:00<00:00, 933.98it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<02:18,  2.87it/s]     12%|#1        | 46/400 [00:00<01:26,  4.09it/s]     22%|##2       | 90/400 [00:00<00:53,  5.82it/s]     34%|###4      | 137/400 [00:00<00:31,  8.27it/s]     44%|####3     | 175/400 [00:00<00:19, 11.71it/s]     56%|#####5    | 222/400 [00:00<00:10, 16.54it/s]     67%|######7   | 268/400 [00:00<00:05, 23.27it/s]     78%|#######8  | 314/400 [00:01<00:02, 32.53it/s]     89%|########8 | 355/400 [00:01<00:01, 44.94it/s]    100%|##########| 400/400 [00:01<00:00, 319.40it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      8%|8         | 81/1000 [00:00<00:01, 805.73it/s]     18%|#8        | 183/1000 [00:00<00:00, 859.05it/s]     28%|##8       | 285/1000 [00:00<00:00, 900.36it/s]     39%|###8      | 389/1000 [00:00<00:00, 937.17it/s]     48%|####8     | 483/1000 [00:00<00:00, 936.13it/s]     57%|#####7    | 574/1000 [00:00<00:00, 922.51it/s]     67%|######6   | 668/1000 [00:00<00:00, 926.86it/s]     76%|#######6  | 764/1000 [00:00<00:00, 934.54it/s]     86%|########6 | 862/1000 [00:00<00:00, 947.60it/s]     95%|#########5| 954/1000 [00:01<00:00, 936.36it/s]    100%|##########| 1000/1000 [00:01<00:00, 947.93it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<05:21,  3.11it/s]      4%|4         | 44/1000 [00:00<03:35,  4.43it/s]      9%|9         | 92/1000 [00:00<02:24,  6.30it/s]     13%|#3        | 133/1000 [00:00<01:37,  8.94it/s]     17%|#7        | 174/1000 [00:00<01:05, 12.65it/s]     22%|##1       | 219/1000 [00:00<00:43, 17.85it/s]     27%|##6       | 267/1000 [00:00<00:29, 25.10it/s]     31%|###1      | 313/1000 [00:01<00:19, 35.03it/s]     36%|###5      | 356/1000 [00:01<00:13, 48.33it/s]     40%|###9      | 398/1000 [00:01<00:09, 65.63it/s]     44%|####4     | 444/1000 [00:01<00:06, 88.34it/s]     49%|####8     | 487/1000 [00:01<00:04, 115.87it/s]     54%|#####3    | 537/1000 [00:01<00:03, 150.37it/s]     58%|#####8    | 585/1000 [00:01<00:02, 189.08it/s]     63%|######3   | 631/1000 [00:01<00:01, 227.50it/s]     68%|######7   | 676/1000 [00:01<00:01, 265.01it/s]     72%|#######2  | 721/1000 [00:01<00:00, 289.24it/s]     77%|#######6  | 767/1000 [00:02<00:00, 325.02it/s]     81%|########1 | 814/1000 [00:02<00:00, 356.74it/s]     86%|########5 | 859/1000 [00:02<00:00, 371.88it/s]     90%|######### | 905/1000 [00:02<00:00, 394.16it/s]     95%|#########4| 949/1000 [00:02<00:00, 402.38it/s]    100%|#########9| 995/1000 [00:02<00:00, 415.68it/s]    100%|##########| 1000/1000 [00:02<00:00, 383.87it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      8%|8         | 81/1000 [00:00<00:01, 808.12it/s]     18%|#7        | 179/1000 [00:00<00:00, 850.89it/s]     27%|##7       | 272/1000 [00:00<00:00, 872.83it/s]     36%|###6      | 365/1000 [00:00<00:00, 887.83it/s]     46%|####5     | 459/1000 [00:00<00:00, 902.70it/s]     55%|#####5    | 553/1000 [00:00<00:00, 911.91it/s]     65%|######4   | 647/1000 [00:00<00:00, 919.47it/s]     73%|#######3  | 733/1000 [00:00<00:00, 866.19it/s]     82%|########2 | 823/1000 [00:00<00:00, 874.13it/s]     92%|#########1| 916/1000 [00:01<00:00, 887.85it/s]    100%|##########| 1000/1000 [00:01<00:00, 907.75it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<04:51,  3.43it/s]      4%|4         | 45/1000 [00:00<03:15,  4.88it/s]     10%|9         | 97/1000 [00:00<02:10,  6.95it/s]     15%|#5        | 150/1000 [00:00<01:26,  9.87it/s]     20%|##        | 200/1000 [00:00<00:57, 13.97it/s]     25%|##5       | 254/1000 [00:00<00:37, 19.74it/s]     31%|###       | 309/1000 [00:00<00:24, 27.77it/s]     36%|###5      | 359/1000 [00:00<00:16, 38.75it/s]     41%|####1     | 414/1000 [00:01<00:10, 53.73it/s]     46%|####6     | 464/1000 [00:01<00:07, 73.31it/s]     52%|#####2    | 522/1000 [00:01<00:04, 99.31it/s]     57%|#####7    | 574/1000 [00:01<00:03, 130.63it/s]     63%|######2   | 626/1000 [00:01<00:02, 167.68it/s]     68%|######7   | 677/1000 [00:01<00:01, 201.99it/s]     72%|#######2  | 725/1000 [00:01<00:01, 240.89it/s]     77%|#######7  | 772/1000 [00:01<00:00, 280.01it/s]     82%|########2 | 820/1000 [00:01<00:00, 319.29it/s]     87%|########6 | 867/1000 [00:02<00:00, 349.68it/s]     92%|#########1| 918/1000 [00:02<00:00, 384.49it/s]     97%|#########6| 966/1000 [00:02<00:00, 405.70it/s]    100%|##########| 1000/1000 [00:02<00:00, 426.97it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      8%|7         | 79/1000 [00:00<00:01, 787.66it/s]     18%|#8        | 183/1000 [00:00<00:00, 848.93it/s]     28%|##8       | 283/1000 [00:00<00:00, 887.76it/s]     38%|###7      | 377/1000 [00:00<00:00, 902.75it/s]     48%|####7     | 478/1000 [00:00<00:00, 930.63it/s]     58%|#####7    | 577/1000 [00:00<00:00, 946.38it/s]     68%|######7   | 677/1000 [00:00<00:00, 959.54it/s]     77%|#######7  | 770/1000 [00:00<00:00, 949.58it/s]     86%|########6 | 864/1000 [00:00<00:00, 945.42it/s]     96%|#########6| 960/1000 [00:01<00:00, 948.24it/s]    100%|##########| 1000/1000 [00:01<00:00, 956.63it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<05:00,  3.32it/s]      5%|4         | 48/1000 [00:00<03:21,  4.73it/s]      9%|9         | 93/1000 [00:00<02:14,  6.73it/s]     14%|#4        | 140/1000 [00:00<01:29,  9.56it/s]     19%|#8        | 188/1000 [00:00<00:59, 13.54it/s]     24%|##3       | 236/1000 [00:00<00:39, 19.11it/s]     29%|##8       | 287/1000 [00:00<00:26, 26.85it/s]     33%|###3      | 330/1000 [00:01<00:17, 37.31it/s]     37%|###7      | 374/1000 [00:01<00:12, 51.41it/s]     42%|####1     | 418/1000 [00:01<00:08, 69.93it/s]     47%|####6     | 470/1000 [00:01<00:05, 94.44it/s]     52%|#####1    | 517/1000 [00:01<00:03, 124.08it/s]     56%|#####6    | 563/1000 [00:01<00:02, 158.83it/s]     61%|######1   | 610/1000 [00:01<00:01, 198.12it/s]     66%|######5   | 658/1000 [00:01<00:01, 239.73it/s]     71%|#######   | 708/1000 [00:01<00:01, 284.06it/s]     76%|#######5  | 757/1000 [00:01<00:00, 324.02it/s]     80%|########  | 805/1000 [00:02<00:00, 354.01it/s]     85%|########5 | 852/1000 [00:02<00:00, 373.65it/s]     90%|########9 | 898/1000 [00:02<00:00, 393.86it/s]     94%|#########4| 944/1000 [00:02<00:00, 410.98it/s]     99%|#########9| 990/1000 [00:02<00:00, 423.88it/s]    100%|##########| 1000/1000 [00:02<00:00, 406.74it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     25%|##5       | 100/400 [00:00<00:00, 998.80it/s]     53%|#####2    | 211/400 [00:00<00:00, 1026.84it/s]     80%|########  | 322/400 [00:00<00:00, 1050.32it/s]    100%|##########| 400/400 [00:00<00:00, 1089.28it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:40,  3.96it/s]     26%|##6       | 106/400 [00:00<00:52,  5.64it/s]     53%|#####2    | 211/400 [00:00<00:23,  8.04it/s]     80%|########  | 320/400 [00:00<00:06, 11.45it/s]    100%|##########| 400/400 [00:00<00:00, 645.53it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     24%|##4       | 97/400 [00:00<00:00, 965.63it/s]     53%|#####3    | 213/400 [00:00<00:00, 1014.91it/s]     80%|########  | 322/400 [00:00<00:00, 1036.07it/s]    100%|##########| 400/400 [00:00<00:00, 1073.61it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:38,  4.04it/s]     29%|##9       | 116/400 [00:00<00:49,  5.76it/s]     59%|#####8    | 235/400 [00:00<00:20,  8.22it/s]     84%|########4 | 338/400 [00:00<00:05, 11.70it/s]    100%|##########| 400/400 [00:00<00:00, 659.14it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     25%|##5       | 100/400 [00:00<00:00, 998.09it/s]     55%|#####4    | 219/400 [00:00<00:00, 1048.03it/s]     83%|########2 | 332/400 [00:00<00:00, 1065.31it/s]    100%|##########| 400/400 [00:00<00:00, 1098.47it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:38,  4.06it/s]     28%|##8       | 114/400 [00:00<00:49,  5.79it/s]     57%|#####7    | 230/400 [00:00<00:20,  8.25it/s]     86%|########5 | 342/400 [00:00<00:04, 11.75it/s]    100%|##########| 400/400 [00:00<00:00, 668.20it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      7%|7         | 70/1000 [00:00<00:01, 697.46it/s]     16%|#6        | 161/1000 [00:00<00:01, 748.09it/s]     25%|##5       | 253/1000 [00:00<00:00, 790.74it/s]     35%|###4      | 347/1000 [00:00<00:00, 829.86it/s]     44%|####4     | 443/1000 [00:00<00:00, 864.52it/s]     54%|#####3    | 535/1000 [00:00<00:00, 877.39it/s]     62%|######2   | 625/1000 [00:00<00:00, 883.61it/s]     72%|#######1  | 716/1000 [00:00<00:00, 890.30it/s]     80%|########  | 802/1000 [00:00<00:00, 878.07it/s]     90%|########9 | 898/1000 [00:01<00:00, 900.22it/s]     99%|#########9| 991/1000 [00:01<00:00, 907.98it/s]    100%|##########| 1000/1000 [00:01<00:00, 896.11it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<04:12,  3.95it/s]     11%|#1        | 110/1000 [00:00<02:37,  5.64it/s]     22%|##2       | 224/1000 [00:00<01:36,  8.04it/s]     34%|###4      | 341/1000 [00:00<00:57, 11.45it/s]     46%|####5     | 455/1000 [00:00<00:33, 16.29it/s]     56%|#####6    | 565/1000 [00:00<00:18, 23.12it/s]     68%|######7   | 676/1000 [00:00<00:09, 32.73it/s]     79%|#######9  | 792/1000 [00:00<00:04, 46.20it/s]     90%|########9 | 896/1000 [00:01<00:01, 64.76it/s]    100%|##########| 1000/1000 [00:01<00:00, 870.90it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      9%|9         | 92/1000 [00:00<00:00, 912.93it/s]     20%|##        | 203/1000 [00:00<00:00, 963.16it/s]     32%|###2      | 322/1000 [00:00<00:00, 1021.25it/s]     44%|####4     | 441/1000 [00:00<00:00, 1064.24it/s]     56%|#####6    | 560/1000 [00:00<00:00, 1098.69it/s]     68%|######8   | 681/1000 [00:00<00:00, 1126.90it/s]     79%|#######9  | 790/1000 [00:00<00:00, 1115.40it/s]     90%|######### | 901/1000 [00:00<00:00, 1111.18it/s]    100%|##########| 1000/1000 [00:00<00:00, 1112.73it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<04:12,  3.95it/s]     11%|#1        | 112/1000 [00:00<02:37,  5.63it/s]     22%|##2       | 224/1000 [00:00<01:36,  8.03it/s]     33%|###3      | 334/1000 [00:00<00:58, 11.44it/s]     44%|####4     | 443/1000 [00:00<00:34, 16.27it/s]     55%|#####5    | 554/1000 [00:00<00:19, 23.09it/s]     67%|######7   | 671/1000 [00:00<00:10, 32.71it/s]     78%|#######7  | 778/1000 [00:00<00:04, 46.12it/s]     88%|########8 | 880/1000 [00:01<00:01, 64.64it/s]     99%|#########9| 992/1000 [00:01<00:00, 90.11it/s]    100%|##########| 1000/1000 [00:01<00:00, 858.04it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      8%|8         | 82/1000 [00:00<00:01, 816.91it/s]     18%|#7        | 177/1000 [00:00<00:00, 852.51it/s]     27%|##7       | 270/1000 [00:00<00:00, 871.91it/s]     36%|###6      | 360/1000 [00:00<00:00, 878.46it/s]     45%|####5     | 453/1000 [00:00<00:00, 890.71it/s]     55%|#####4    | 548/1000 [00:00<00:00, 907.21it/s]     64%|######4   | 644/1000 [00:00<00:00, 920.12it/s]     74%|#######4  | 740/1000 [00:00<00:00, 931.36it/s]     83%|########3 | 833/1000 [00:00<00:00, 930.27it/s]     92%|#########2| 924/1000 [00:01<00:00, 922.18it/s]    100%|##########| 1000/1000 [00:01<00:00, 914.25it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<04:21,  3.82it/s]     11%|#1        | 112/1000 [00:00<02:42,  5.45it/s]     22%|##2       | 225/1000 [00:00<01:39,  7.77it/s]     33%|###3      | 331/1000 [00:00<01:00, 11.06it/s]     44%|####4     | 443/1000 [00:00<00:35, 15.74it/s]     56%|#####6    | 564/1000 [00:00<00:19, 22.36it/s]     68%|######8   | 681/1000 [00:00<00:10, 31.68it/s]     80%|########  | 800/1000 [00:00<00:04, 44.75it/s]     91%|######### | 908/1000 [00:01<00:01, 62.81it/s]    100%|##########| 1000/1000 [00:01<00:00, 870.96it/s]
    /home/vrael/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())






|


.. code-block:: default


    import matplotlib.pyplot as plt

    import symjax
    import symjax.tensor as T
    from symjax.nn import optimizers
    import numpy as np
    from tqdm import tqdm


    BS = 1000
    D = 500
    X = np.random.randn(BS, D).astype("float32")
    Y = X.dot(np.random.randn(D, 1).astype("float32")) + 2


    def TF1(x, y, N, lr, model, preallocate=False):
        import tensorflow.compat.v1 as tf

        tf.compat.v1.disable_v2_behavior()
        tf.reset_default_graph()

        tf_input = tf.placeholder(dtype=tf.float32, shape=[BS, D])
        tf_output = tf.placeholder(dtype=tf.float32, shape=[BS, 1])

        np.random.seed(0)

        tf_W = tf.Variable(np.random.randn(D, 1).astype("float32"))
        tf_b = tf.Variable(
            np.random.randn(
                1,
            ).astype("float32")
        )

        tf_loss = tf.reduce_mean((tf.matmul(tf_input, tf_W) + tf_b - tf_output) ** 2)
        if model == "SGD":
            train_op = tf.train.GradientDescentOptimizer(lr).minimize(tf_loss)
        elif model == "Adam":
            train_op = tf.train.AdamOptimizer(lr).minimize(tf_loss)

        # initialize session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        losses = []
        for i in tqdm(range(N)):
            losses.append(
                sess.run([tf_loss, train_op], feed_dict={tf_input: x, tf_output: y})[0]
            )

        return losses


    def TF_EMA(X):
        import tensorflow.compat.v1 as tf

        tf.compat.v1.disable_v2_behavior()
        tf.reset_default_graph()
        x = tf.placeholder("float32")
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        op = ema.apply([x])
        out = ema.average(x)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer(), feed_dict={x: X[0]})

        outputs = []
        for i in range(len(X)):
            sess.run(op, feed_dict={x: X[i]})
            outputs.append(sess.run(out))
        return outputs


    def SJ_EMA(X, debias=True):
        symjax.current_graph().reset()
        x = T.Placeholder((), "float32", name="x")
        value = symjax.nn.schedules.ExponentialMovingAverage(x, 0.9, debias=debias)[0]
        print(x, value)
        print(symjax.current_graph().roots(value))
        train = symjax.function(x, outputs=value, updates=symjax.get_updates())
        outputs = []
        for i in range(len(X)):
            outputs.append(train(X[i]))
        return outputs


    def SJ(x, y, N, lr, model, preallocate=False):
        symjax.current_graph().reset()
        sj_input = T.Placeholder(dtype=np.float32, shape=[BS, D])
        sj_output = T.Placeholder(dtype=np.float32, shape=[BS, 1])

        np.random.seed(0)

        sj_W = T.Variable(np.random.randn(D, 1).astype("float32"))
        sj_b = T.Variable(
            np.random.randn(
                1,
            ).astype("float32")
        )

        sj_loss = ((sj_input.dot(sj_W) + sj_b - sj_output) ** 2).mean()

        if model == "SGD":
            optimizers.SGD(sj_loss, lr)
        elif model == "Adam":
            optimizers.Adam(sj_loss, lr)
        train = symjax.function(
            sj_input, sj_output, outputs=sj_loss, updates=symjax.get_updates()
        )

        losses = []
        for i in tqdm(range(N)):
            losses.append(train(x, y))

        return losses


    sample = np.random.randn(100)

    plt.figure()
    plt.plot(sample, label="Original signal", alpha=0.5)
    plt.plot(TF_EMA(sample), c="orange", label="TF ema", linewidth=2, alpha=0.5)
    plt.plot(SJ_EMA(sample), c="green", label="SJ ema (biased)", linewidth=2, alpha=0.5)
    plt.plot(
        SJ_EMA(sample, False),
        c="green",
        linestyle="--",
        label="SJ ema (unbiased)",
        linewidth=2,
        alpha=0.5,
    )
    plt.legend()


    plt.figure()
    Ns = [400, 1000]
    lrs = [0.001, 0.01, 0.1]
    colors = ["r", "b", "g"]
    for k, N in enumerate(Ns):
        plt.subplot(1, len(Ns), 1 + k)
        for c, lr in enumerate(lrs):
            loss = TF1(X, Y, N, lr, "Adam")
            plt.plot(loss, c=colors[c], linestyle="-", alpha=0.5)
            loss = SJ(X, Y, N, lr, "Adam")
            plt.plot(loss, c=colors[c], linestyle="--", alpha=0.5, linewidth=2)
            plt.title("lr:" + str(lr))
    plt.suptitle("Adam Optimization quadratic loss (-:TF, --:SJ)")


    plt.figure()
    Ns = [400, 1000]
    lrs = [0.001, 0.01, 0.1]
    colors = ["r", "b", "g"]
    for k, N in enumerate(Ns):
        plt.subplot(1, len(Ns), 1 + k)
        for c, lr in enumerate(lrs):
            loss = TF1(X, Y, N, lr, "SGD")
            plt.plot(loss, c=colors[c], linestyle="-", alpha=0.5)
            loss = SJ(X, Y, N, lr, "SGD")
            plt.plot(loss, c=colors[c], linestyle="--", alpha=0.5, linewidth=2)
            plt.title("lr:" + str(lr))
            plt.xlabel("steps")
    plt.suptitle("GD Optimization quadratic loss (-:TF, --:SJ)")
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  27.428 seconds)


.. _sphx_glr_download_auto_examples_plot_compare_adam.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_compare_adam.py <plot_compare_adam.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_compare_adam.ipynb <plot_compare_adam.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
