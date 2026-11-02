# Segunda corrida (reproducción)

Mismos detectores, cascada y análisis del Hito 2, reentrenados con el mismo código y configuraciones
en otro equipo (Windows, GPU NVIDIA RTX 4060). Respaldan la columna "Repetido" de la tabla de
reproducibilidad del informe. Los resultados oficiales del informe están en `results/det/`.

- `<detector>/metrics_test.json` y `history.csv`: métricas en test y curvas por época de cada detector.
- `cascade/metrics_test.json`: cascada con el detector de manos de esta corrida.
- `analysis/`: análisis por condición de esta corrida.
- `log_windows.txt`: salida de la consola durante el entrenamiento y la evaluación.
