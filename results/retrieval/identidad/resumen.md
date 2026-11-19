# Hito 3 · ¿Gesto o identidad? (análisis dentro de test, sin pares de la misma foto)

|                                                      |   imagenet |   cnn_residual |   resnet18_ft |
|:-----------------------------------------------------|-----------:|---------------:|--------------:|
| sim media: mismo gesto, otra persona                 |     0.6198 |         0.9326 |        0.9715 |
| sim media: misma persona, otro gesto                 |     0.6163 |         0.5656 |        0.5595 |
| sim media: misma persona, mismo gesto                |     0.6574 |         0.9462 |        0.9701 |
| sim media: otra persona, otro gesto                  |     0.5756 |         0.5683 |        0.5611 |
| top10: mismo gesto, otra persona                     |     0.4881 |         0.9847 |        0.9905 |
| azar: mismo gesto, otra persona                      |     0.0907 |         0.0907 |        0.0907 |
| top10: misma persona, otro gesto                     |     0.0295 |         0.0001 |        0      |
| azar: misma persona, otro gesto                      |     0.0012 |         0.0012 |        0.0012 |
| top10: misma persona, mismo gesto                    |     0.0045 |         0.003  |        0.0043 |
| azar: misma persona, mismo gesto                     |     0      |         0      |        0      |
| top10: otra persona, otro gesto                      |     0.478  |         0.0122 |        0.0051 |
| azar: otra persona, otro gesto                       |     0.908  |         0.908  |        0.908  |
| percentil medio de la misma persona (50% = azar)     |     0.6601 |         0.4961 |        0.4943 |
| consultas con la misma persona en top10 (otro gesto) |     0.2233 |         0.0011 |        0.0004 |
