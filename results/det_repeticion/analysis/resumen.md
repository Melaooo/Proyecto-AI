# Análisis por condición (test)

|                                                      |   mAP50 directo |   mAP50 cascada |   recall_gesto directo |   recall_gesto cascada |   recall_localizacion directo |   recall_localizacion cascada |
|:-----------------------------------------------------|----------------:|----------------:|-----------------------:|-----------------------:|------------------------------:|------------------------------:|
| ('global', 'todo')                                   |           0.933 |           0.968 |                  0.913 |                  0.942 |                         0.937 |                         0.946 |
| ('distancia a la cámara', 'lejos (<83 px)')          |           0.83  |           0.938 |                  0.828 |                  0.89  |                         0.868 |                         0.895 |
| ('distancia a la cámara', 'medio (83-117 px)')       |           0.969 |           0.988 |                  0.957 |                  0.973 |                         0.976 |                         0.974 |
| ('distancia a la cámara', 'cerca (>117 px)')         |           0.969 |           0.98  |                  0.964 |                  0.968 |                         0.975 |                         0.974 |
| ('una / dos manos', '1 mano')                        |           0.956 |           0.982 |                  0.949 |                  0.973 |                         0.972 |                         0.978 |
| ('una / dos manos', '2 manos')                       |           0.89  |           0.955 |                  0.846 |                  0.881 |                         0.87  |                         0.885 |
| ('fondo: brillo', 'bajo (0.03-0.46)')                |           0.927 |           0.968 |                  0.908 |                  0.933 |                         0.934 |                         0.937 |
| ('fondo: brillo', 'medio (0.46-0.56)')               |           0.929 |           0.964 |                  0.903 |                  0.939 |                         0.927 |                         0.942 |
| ('fondo: brillo', 'alto (0.56-0.85)')                |           0.946 |           0.972 |                  0.928 |                  0.953 |                         0.951 |                         0.958 |
| ('fondo: contraste', 'bajo (0.04-0.20)')             |           0.933 |           0.972 |                  0.915 |                  0.95  |                         0.942 |                         0.953 |
| ('fondo: contraste', 'medio (0.20-0.24)')            |           0.936 |           0.961 |                  0.912 |                  0.933 |                         0.934 |                         0.938 |
| ('fondo: contraste', 'alto (0.24-0.40)')             |           0.933 |           0.972 |                  0.911 |                  0.942 |                         0.935 |                         0.946 |
| ('fondo: bordes (complejidad)', 'bajo (0.00-0.10)')  |           0.942 |           0.97  |                  0.929 |                  0.952 |                         0.947 |                         0.955 |
| ('fondo: bordes (complejidad)', 'medio (0.10-0.16)') |           0.936 |           0.966 |                  0.914 |                  0.941 |                         0.938 |                         0.946 |
| ('fondo: bordes (complejidad)', 'alto (0.16-0.69)')  |           0.924 |           0.969 |                  0.897 |                  0.932 |                         0.926 |                         0.936 |

## Imágenes con dos manos

| sistema   | mano (imágenes con 2 manos)   |   manos |   perdidas |   clase equivocada |   recall_gesto |   recall_localizacion |
|:----------|:------------------------------|--------:|-----------:|-------------------:|---------------:|----------------------:|
| directo   | mano con gesto                |    1049 |         83 |                 49 |         0.8742 |                0.9209 |
| directo   | mano no_gesture               |    1049 |        190 |                  2 |         0.817  |                0.8189 |
| cascada   | mano con gesto                |    1049 |         60 |                  7 |         0.9361 |                0.9428 |
| cascada   | mano no_gesture               |    1049 |        181 |                  1 |         0.8265 |                0.8275 |
