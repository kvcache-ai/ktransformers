import matplotlib.pyplot as plt
import re

# 原始数据字符串
data_str = """
execute end,m is:2
计算性能: 0.0068 TFLOPS

execute end,m is:4
计算性能: 0.0143 TFLOPS

execute end,m is:6
计算性能: 0.0206 TFLOPS

execute end,m is:8
计算性能: 0.0273 TFLOPS

execute end,m is:10
计算性能: 0.0330 TFLOPS

execute end,m is:12
计算性能: 0.0390 TFLOPS

execute end,m is:14
计算性能: 0.0442 TFLOPS

execute end,m is:16
计算性能: 0.0495 TFLOPS

execute end,m is:18
计算性能: 0.0543 TFLOPS

execute end,m is:20
计算性能: 0.0595 TFLOPS

execute end,m is:22
计算性能: 0.0637 TFLOPS

execute end,m is:24
计算性能: 0.0684 TFLOPS

execute end,m is:26
计算性能: 0.0720 TFLOPS

execute end,m is:28
计算性能: 0.0769 TFLOPS

execute end,m is:30
计算性能: 0.0802 TFLOPS

execute end,m is:32
计算性能: 0.0843 TFLOPS

execute end,m is:34
计算性能: 0.0874 TFLOPS

execute end,m is:36
计算性能: 0.0916 TFLOPS

execute end,m is:38
计算性能: 0.0942 TFLOPS

execute end,m is:40
计算性能: 0.0977 TFLOPS

execute end,m is:42
计算性能: 0.1003 TFLOPS

execute end,m is:44
计算性能: 0.1043 TFLOPS

execute end,m is:46
计算性能: 0.1059 TFLOPS

execute end,m is:48
计算性能: 0.1103 TFLOPS

execute end,m is:50
计算性能: 0.1119 TFLOPS

execute end,m is:52
计算性能: 0.1153 TFLOPS

execute end,m is:54
计算性能: 0.1172 TFLOPS

execute end,m is:56
计算性能: 0.1202 TFLOPS

execute end,m is:58
计算性能: 0.1219 TFLOPS

execute end,m is:60
计算性能: 0.1251 TFLOPS

execute end,m is:62
计算性能: 0.1268 TFLOPS

execute end,m is:64
计算性能: 0.1286 TFLOPS

execute end,m is:66
计算性能: 0.1307 TFLOPS

execute end,m is:68
计算性能: 0.1342 TFLOPS

execute end,m is:70
计算性能: 0.1347 TFLOPS

execute end,m is:72
计算性能: 0.1383 TFLOPS

execute end,m is:74
计算性能: 0.1389 TFLOPS

execute end,m is:76
计算性能: 0.1416 TFLOPS

execute end,m is:78
计算性能: 0.1429 TFLOPS

execute end,m is:80
计算性能: 0.1451 TFLOPS

execute end,m is:82
计算性能: 0.1471 TFLOPS

execute end,m is:84
计算性能: 0.1489 TFLOPS

execute end,m is:86
计算性能: 0.1499 TFLOPS

execute end,m is:88
计算性能: 0.1519 TFLOPS

execute end,m is:90
计算性能: 0.1525 TFLOPS

execute end,m is:92
计算性能: 0.1544 TFLOPS

execute end,m is:94
计算性能: 0.1560 TFLOPS

execute end,m is:96
计算性能: 0.1583 TFLOPS

execute end,m is:98
计算性能: 0.1579 TFLOPS

execute end,m is:100
计算性能: 0.1600 TFLOPS

execute end,m is:102
计算性能: 0.1611 TFLOPS

execute end,m is:104
计算性能: 0.1630 TFLOPS

execute end,m is:106
计算性能: 0.1644 TFLOPS

execute end,m is:108
计算性能: 0.1669 TFLOPS

execute end,m is:110
计算性能: 0.1667 TFLOPS

execute end,m is:112
计算性能: 0.1687 TFLOPS

execute end,m is:114
计算性能: 0.1685 TFLOPS

execute end,m is:116
计算性能: 0.1712 TFLOPS

execute end,m is:118
计算性能: 0.1712 TFLOPS

execute end,m is:120
计算性能: 0.1733 TFLOPS

execute end,m is:122
计算性能: 0.1730 TFLOPS

execute end,m is:124
计算性能: 0.1753 TFLOPS

execute end,m is:126
计算性能: 0.1757 TFLOPS

execute end,m is:128
计算性能: 0.1767 TFLOPS

execute end,m is:130
计算性能: 0.1783 TFLOPS

execute end,m is:132
计算性能: 0.1792 TFLOPS

execute end,m is:134
计算性能: 0.1794 TFLOPS

execute end,m is:136
计算性能: 0.1821 TFLOPS

execute end,m is:138
计算性能: 0.1810 TFLOPS

execute end,m is:140
计算性能: 0.1844 TFLOPS

execute end,m is:142
计算性能: 0.1840 TFLOPS

execute end,m is:144
计算性能: 0.1853 TFLOPS

execute end,m is:146
计算性能: 0.1860 TFLOPS

execute end,m is:148
计算性能: 0.1867 TFLOPS

execute end,m is:150
计算性能: 0.1868 TFLOPS

execute end,m is:152
计算性能: 0.1882 TFLOPS

execute end,m is:154
计算性能: 0.1880 TFLOPS

execute end,m is:156
计算性能: 0.1900 TFLOPS

execute end,m is:158
计算性能: 0.1895 TFLOPS

execute end,m is:160
计算性能: 0.1921 TFLOPS

execute end,m is:162
计算性能: 0.1922 TFLOPS

execute end,m is:164
计算性能: 0.1937 TFLOPS

execute end,m is:166
计算性能: 0.1935 TFLOPS

execute end,m is:168
计算性能: 0.1934 TFLOPS

execute end,m is:170
计算性能: 0.1945 TFLOPS

execute end,m is:172
计算性能: 0.1961 TFLOPS

execute end,m is:174
计算性能: 0.1952 TFLOPS

execute end,m is:176
计算性能: 0.1962 TFLOPS

execute end,m is:178
计算性能: 0.1977 TFLOPS

execute end,m is:180
计算性能: 0.1980 TFLOPS

execute end,m is:182
计算性能: 0.1985 TFLOPS

execute end,m is:184
计算性能: 0.1993 TFLOPS

execute end,m is:186
计算性能: 0.1995 TFLOPS

execute end,m is:188
计算性能: 0.2007 TFLOPS

execute end,m is:190
计算性能: 0.2012 TFLOPS

execute end,m is:192
计算性能: 0.2024 TFLOPS

execute end,m is:194
计算性能: 0.2011 TFLOPS

execute end,m is:196
计算性能: 0.2037 TFLOPS

execute end,m is:198
计算性能: 0.2026 TFLOPS

execute end,m is:200
计算性能: 0.2044 TFLOPS

execute end,m is:202
计算性能: 0.2044 TFLOPS

execute end,m is:204
计算性能: 0.2052 TFLOPS

execute end,m is:206
计算性能: 0.2057 TFLOPS

execute end,m is:208
计算性能: 0.2061 TFLOPS

execute end,m is:210
计算性能: 0.2064 TFLOPS

execute end,m is:212
计算性能: 0.2074 TFLOPS

execute end,m is:214
计算性能: 0.2075 TFLOPS

execute end,m is:216
计算性能: 0.2082 TFLOPS

execute end,m is:218
计算性能: 0.2083 TFLOPS

execute end,m is:220
计算性能: 0.2091 TFLOPS

execute end,m is:222
计算性能: 0.2096 TFLOPS

execute end,m is:224
计算性能: 0.2097 TFLOPS

execute end,m is:226
计算性能: 0.2098 TFLOPS

execute end,m is:228
计算性能: 0.2107 TFLOPS

execute end,m is:230
计算性能: 0.2104 TFLOPS

execute end,m is:232
计算性能: 0.2118 TFLOPS

execute end,m is:234
计算性能: 0.2121 TFLOPS

execute end,m is:236
计算性能: 0.2125 TFLOPS

execute end,m is:238
计算性能: 0.2128 TFLOPS

execute end,m is:240
计算性能: 0.2133 TFLOPS

execute end,m is:242
计算性能: 0.2136 TFLOPS

execute end,m is:244
计算性能: 0.2137 TFLOPS

execute end,m is:246
计算性能: 0.2139 TFLOPS

execute end,m is:248
计算性能: 0.2150 TFLOPS

execute end,m is:250
计算性能: 0.2153 TFLOPS

execute end,m is:252
计算性能: 0.2160 TFLOPS

execute end,m is:254
计算性能: 0.2156 TFLOPS

execute end,m is:256
计算性能: 0.2169 TFLOPS

execute end,m is:258
计算性能: 0.2161 TFLOPS

execute end,m is:260
计算性能: 0.2175 TFLOPS

execute end,m is:262
计算性能: 0.2172 TFLOPS

execute end,m is:264
计算性能: 0.2175 TFLOPS

execute end,m is:266
计算性能: 0.2181 TFLOPS

execute end,m is:268
计算性能: 0.2189 TFLOPS

execute end,m is:270
计算性能: 0.2193 TFLOPS

execute end,m is:272
计算性能: 0.2201 TFLOPS

execute end,m is:274
计算性能: 0.2198 TFLOPS

execute end,m is:276
计算性能: 0.2195 TFLOPS

execute end,m is:278
计算性能: 0.2205 TFLOPS

execute end,m is:280
计算性能: 0.2212 TFLOPS

execute end,m is:282
计算性能: 0.2210 TFLOPS

execute end,m is:284
计算性能: 0.2210 TFLOPS

execute end,m is:286
计算性能: 0.2215 TFLOPS

execute end,m is:288
计算性能: 0.2225 TFLOPS

execute end,m is:290
计算性能: 0.2227 TFLOPS

execute end,m is:292
计算性能: 0.2234 TFLOPS

execute end,m is:294
计算性能: 0.2227 TFLOPS

execute end,m is:296
计算性能: 0.2242 TFLOPS

execute end,m is:298
计算性能: 0.2230 TFLOPS

execute end,m is:300
计算性能: 0.2232 TFLOPS

execute end,m is:302
计算性能: 0.2227 TFLOPS

execute end,m is:304
计算性能: 0.2234 TFLOPS

execute end,m is:306
计算性能: 0.2226 TFLOPS

execute end,m is:308
计算性能: 0.2239 TFLOPS

execute end,m is:310
计算性能: 0.2239 TFLOPS

execute end,m is:312
计算性能: 0.2249 TFLOPS

execute end,m is:314
计算性能: 0.2245 TFLOPS

execute end,m is:316
计算性能: 0.2254 TFLOPS

execute end,m is:318
计算性能: 0.2251 TFLOPS

execute end,m is:320
计算性能: 0.2262 TFLOPS

execute end,m is:322
计算性能: 0.2256 TFLOPS

execute end,m is:324
计算性能: 0.2262 TFLOPS

execute end,m is:326
计算性能: 0.2259 TFLOPS

execute end,m is:328
计算性能: 0.2265 TFLOPS

execute end,m is:330
计算性能: 0.2266 TFLOPS

execute end,m is:332
计算性能: 0.2275 TFLOPS

execute end,m is:334
计算性能: 0.2275 TFLOPS

execute end,m is:336
计算性能: 0.2280 TFLOPS

execute end,m is:338
计算性能: 0.2275 TFLOPS

execute end,m is:340
计算性能: 0.2281 TFLOPS

execute end,m is:342
计算性能: 0.2284 TFLOPS

execute end,m is:344
计算性能: 0.2288 TFLOPS

execute end,m is:346
计算性能: 0.2288 TFLOPS

execute end,m is:348
计算性能: 0.2295 TFLOPS

execute end,m is:350
计算性能: 0.2292 TFLOPS

execute end,m is:352
计算性能: 0.2300 TFLOPS

execute end,m is:354
计算性能: 0.2299 TFLOPS

execute end,m is:356
计算性能: 0.2303 TFLOPS

execute end,m is:358
计算性能: 0.2301 TFLOPS

execute end,m is:360
计算性能: 0.2307 TFLOPS

execute end,m is:362
计算性能: 0.2303 TFLOPS

execute end,m is:364
计算性能: 0.2312 TFLOPS

execute end,m is:366
计算性能: 0.2307 TFLOPS

execute end,m is:368
计算性能: 0.2316 TFLOPS

execute end,m is:370
计算性能: 0.2310 TFLOPS

execute end,m is:372
计算性能: 0.2318 TFLOPS

execute end,m is:374
计算性能: 0.2319 TFLOPS

execute end,m is:376
计算性能: 0.2320 TFLOPS

execute end,m is:378
计算性能: 0.2323 TFLOPS

execute end,m is:380
计算性能: 0.2328 TFLOPS

execute end,m is:382
计算性能: 0.2326 TFLOPS

execute end,m is:384
计算性能: 0.2328 TFLOPS

execute end,m is:386
计算性能: 0.2330 TFLOPS

execute end,m is:388
计算性能: 0.2334 TFLOPS

execute end,m is:390
计算性能: 0.2337 TFLOPS

execute end,m is:392
计算性能: 0.2336 TFLOPS

execute end,m is:394
计算性能: 0.2332 TFLOPS

execute end,m is:396
计算性能: 0.2341 TFLOPS

execute end,m is:398
计算性能: 0.2334 TFLOPS

execute end,m is:400
计算性能: 0.2347 TFLOPS

execute end,m is:402
计算性能: 0.2349 TFLOPS

execute end,m is:404
计算性能: 0.2350 TFLOPS

execute end,m is:406
计算性能: 0.2347 TFLOPS

execute end,m is:408
计算性能: 0.2353 TFLOPS

execute end,m is:410
计算性能: 0.2350 TFLOPS

execute end,m is:412
计算性能: 0.2356 TFLOPS

execute end,m is:414
计算性能: 0.2354 TFLOPS

execute end,m is:416
计算性能: 0.2357 TFLOPS

execute end,m is:418
计算性能: 0.2357 TFLOPS

execute end,m is:420
计算性能: 0.2361 TFLOPS

execute end,m is:422
计算性能: 0.2361 TFLOPS

execute end,m is:424
计算性能: 0.2364 TFLOPS

execute end,m is:426
计算性能: 0.2360 TFLOPS

execute end,m is:428
计算性能: 0.2372 TFLOPS

execute end,m is:430
计算性能: 0.2364 TFLOPS

execute end,m is:432
计算性能: 0.2369 TFLOPS

execute end,m is:434
计算性能: 0.2369 TFLOPS

execute end,m is:436
计算性能: 0.2372 TFLOPS

execute end,m is:438
计算性能: 0.2370 TFLOPS

execute end,m is:440
计算性能: 0.2377 TFLOPS

execute end,m is:442
计算性能: 0.2374 TFLOPS

execute end,m is:444
计算性能: 0.2382 TFLOPS

execute end,m is:446
计算性能: 0.2379 TFLOPS

execute end,m is:448
计算性能: 0.2385 TFLOPS

execute end,m is:450
计算性能: 0.2377 TFLOPS

execute end,m is:452
计算性能: 0.2385 TFLOPS

execute end,m is:454
计算性能: 0.2384 TFLOPS

execute end,m is:456
计算性能: 0.2389 TFLOPS

execute end,m is:458
计算性能: 0.2319 TFLOPS

execute end,m is:460
计算性能: 0.2386 TFLOPS

execute end,m is:462
计算性能: 0.2386 TFLOPS

execute end,m is:464
计算性能: 0.2389 TFLOPS

execute end,m is:466
计算性能: 0.2393 TFLOPS

execute end,m is:468
计算性能: 0.2393 TFLOPS

execute end,m is:470
计算性能: 0.2389 TFLOPS

execute end,m is:472
计算性能: 0.2393 TFLOPS

execute end,m is:474
计算性能: 0.2395 TFLOPS

execute end,m is:476
计算性能: 0.2399 TFLOPS

execute end,m is:478
计算性能: 0.2400 TFLOPS

execute end,m is:480
计算性能: 0.2400 TFLOPS

execute end,m is:482
计算性能: 0.2397 TFLOPS

execute end,m is:484
计算性能: 0.2407 TFLOPS

execute end,m is:486
计算性能: 0.2400 TFLOPS

execute end,m is:488
计算性能: 0.2407 TFLOPS

execute end,m is:490
计算性能: 0.2404 TFLOPS

execute end,m is:492
计算性能: 0.2411 TFLOPS

execute end,m is:494
计算性能: 0.2409 TFLOPS

execute end,m is:496
计算性能: 0.2407 TFLOPS

execute end,m is:498
计算性能: 0.2412 TFLOPS

execute end,m is:500
计算性能: 0.2418 TFLOPS

execute end,m is:502
计算性能: 0.2416 TFLOPS

execute end,m is:504
计算性能: 0.2418 TFLOPS

execute end,m is:506
计算性能: 0.2416 TFLOPS

execute end,m is:508
计算性能: 0.2421 TFLOPS

execute end,m is:510
计算性能: 0.2419 TFLOPS

execute end,m is:512
计算性能: 0.2423 TFLOPS
"""

# 使用正则表达式提取 m 和 TFLOPS 值
m_values = list(map(int, re.findall(r'm is:(\d+)', data_str)))
tflops_values = list(map(float, re.findall(r'计算性能: ([\d.]+) TFLOPS', data_str)))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(m_values, tflops_values, marker='o', linestyle='-', color='blue')
plt.title('m * k with k * n (k=7168 n=512) ')
plt.xlabel('m')
plt.ylabel('Tflops')
plt.grid(True)
plt.tight_layout()

# 保存图表为文件
plt.savefig('performance_plot.png')  # 保存为 PNG 格式
# plt.savefig('performance_plot.pdf')  # 或保存为 PDF 格式
