# Hito 3 · Sistema completo (test, 6049 manos, galería = train)

| sistema              |   manos |    mAP |    R@1 |    R@5 |   P@10 |   kNN10_acc |
|:---------------------|--------:|-------:|-------:|-------:|-------:|------------:|
| techo: caja real     |    6049 | 0.9958 | 0.9959 | 0.9974 | 0.9954 |      0.9955 |
| detector + retrieval |    6049 | 0.9547 | 0.9542 | 0.956  | 0.9541 |      0.954  |

Detector de manos con umbral 0.25 (elegido en val, Hito 2): 6150 detecciones, 344 falsos positivos, 243 manos perdidas.

Similitud top-1 media: correctas 0.990, falsos positivos 0.937

## Por tamaño de mano

| tamano            | sistema   |   manos |    mAP |    R@1 |    R@5 |   P@10 |   kNN10_acc |   perdidas |
|:------------------|:----------|--------:|-------:|-------:|-------:|-------:|------------:|-----------:|
| lejos (<83 px)    | techo     |    2172 | 0.9948 | 0.9954 | 0.9968 | 0.9943 |      0.994  |        183 |
| lejos (<83 px)    | sistema   |    2172 | 0.9096 | 0.9088 | 0.9107 | 0.909  |      0.9088 |        183 |
| medio (83-117 px) | techo     |    1981 | 0.9987 | 0.9985 | 0.999  | 0.9984 |      0.9985 |         38 |
| medio (83-117 px) | sistema   |    1981 | 0.9786 | 0.9788 | 0.9788 | 0.978  |      0.9788 |         38 |
| cerca (>117 px)   | techo     |    1896 | 0.9938 | 0.9937 | 0.9963 | 0.9935 |      0.9942 |         22 |
| cerca (>117 px)   | sistema   |    1896 | 0.9813 | 0.9805 | 0.9842 | 0.9806 |      0.98   |         22 |

## Por IoU de la caja detectada (solo manos encontradas)

| IoU caja detectada   |   manos |   R@1 sistema |   R@1 techo (caja real) |   P@10 sistema |   P@10 techo |
|:---------------------|--------:|--------------:|------------------------:|---------------:|-------------:|
| 0.50-0.70            |     485 |        0.9711 |                  0.9918 |         0.9707 |       0.9903 |
| 0.70-0.85            |    2123 |        0.9939 |                  0.9981 |         0.9949 |       0.9979 |
| 0.85-1.00            |    3198 |        0.9978 |                  0.9972 |         0.9969 |       0.9966 |
