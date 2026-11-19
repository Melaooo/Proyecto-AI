# Hito 3 — Guía para Tomás (y para su Claude)

> **Contexto para Claude:** este archivo lo dejó la "Persona 1" (Javier) al terminar su parte del Hito 3.
> Explica qué se hizo, qué falta (la parte de la "Persona 2", Tomás) y cómo hacerlo. Léelo completo
> antes de tocar código. Es el mismo formato que `HITO2_PARA_JAVIER.md`.

## 0. Qué pide el enunciado (PDF de proyectos, tema 24 + pauta general)

> Hito 3 — Retrieval + sistema final: *Retrieval de gestos detectados sobre una galería separada por usuario.
> Evaluar Top-K y verificar que el embedding represente el gesto más que la identidad de la persona.*
> Demostración final esperada: *Webcam/imagen -> mano localizada -> gesto -> ejemplos similares o comando asociado.*

Pauta general: *a partir de la región detectada, extraer un embedding con el encoder y buscar los vecinos más
cercanos en una galería. La entrega final debe integrar imagen -> detección -> crop -> embedding -> Top-K.
Incluir consultas reales, casos correctos/fallidos y evaluación cuantitativa* (Recall@K, Precision@K, mAP de
retrieval, Top-K accuracy). **La galería y las queries deben quedar separadas** (no vale buscar la propia imagen).

## 1. Qué está hecho (Persona 1)

Retrieval con **recortes de las cajas reales** (GT), es decir, el "techo" del sistema, más el análisis de identidad.
Todo es KNN sobre embeddings (la misma idea del lab `Lab_Introduccion_Clasificacion_KNN`) + PCA para visualizar
(como en los labs). No se entrenó nada nuevo: se reutilizan los encoders del Hito 1.

| Pieza | Archivo |
|---|---|
| Embeddings, galería, búsqueda Top-K (`search`), métricas, comparación de encoders y figuras | `src/retrieval.py` |
| ¿Gesto o identidad? (análisis dentro de test) | `src/retrieval_identidad.py` |
| Resultados | `results/retrieval/` (tabla en `comparison.md`, identidad en `identidad/resumen.md`) |
| Logs | `logs/retrieval/retrieval.log`, `logs/retrieval/identidad.log` |

**Protocolo:**
- **Galería = recortes de train** (16 500 manos, 7 720 personas). **Consultas = val** (para elegir) y **test**
  (se reporta una vez). Los splits son por usuario, así que nadie de las consultas está en la galería
  (hay un `assert` que lo comprueba).
- Embedding = salida del encoder **normalizada a largo 1** → similitud coseno = producto punto (como `Q K^T`
  de la clase de atención).
- Relevante = vecino con el **mismo gesto**. Métricas por consulta:
  `R@K` (≥1 relevante en el Top-K, = Top-K accuracy), `P@K` (fracción de relevantes en el Top-K),
  `mAP` (AP sobre el ranking **completo** de la galería) y `kNN10` (gesto por votación de 10 vecinos).
- El encoder se **elige por mAP en val**: gana `resnet18_ft`.

**Comandos:**

```bash
python -m src.retrieval                 # unos minutos con GPU; prueba rápida: --limit 500
python -m src.retrieval_identidad       # usa los embeddings que dejó el anterior
```

Los embeddings quedan en `data/embeddings/` (128 MB, **no subir a git**, se regeneran solos). La primera vez se
descargan los pesos ImageNet de ResNet18 de torchvision (~45 MB) para el encoder `imagenet`.

## 2. Resultados de la Persona 1

### 2.1 Comparación de encoders (test, 5 500 consultas contra galería de 16 500)

| Encoder | Dim | mAP | R@1 | R@5 | P@10 | kNN10 F1 |
|---|---|---|---|---|---|---|
| `imagenet` (ResNet18 ImageNet, sin ajustar) | 512 | 0.223 | 0.604 | 0.887 | 0.536 | 0.694 |
| `landmarks` (geometría, sin CNN; 2.3 % sin landmarks = fallo) | 42 | 0.519 | 0.961 | 0.971 | 0.957 | 0.974 |
| `cnn_residual` (CNN propia del Hito 1) | 256 | 0.986 | 0.990 | 0.996 | 0.990 | 0.992 |
| **`resnet18_ft`** (encoder elegido en el Hito 1) | 512 | **0.996** | **0.996** | **0.997** | **0.995** | **0.996** |

Val da lo mismo (diferencias < 0.01). Figuras: `results/retrieval/precision_vs_k.png`,
`results/retrieval/pca_embeddings.png`, `results/retrieval/<encoder>/topk_ejemplos_test.png`.

Lecturas:
- **El encoder del Hito 1 sirve para retrieval sin reentrenarlo:** mAP 0.996 vs 0.223 de ImageNet. Con ImageNet
  el vecino más cercano es del mismo gesto solo el 60 % de las veces. En el PCA, ImageNet es una nube mezclada
  y `resnet18_ft` forma un grupo por gesto. Es lo mismo que pasó con el linear probe del Hito 1 (0.76): las
  features de ImageNet no separan "uno vs dos dedos".
- **kNN10 con `resnet18_ft` (F1 0.996) iguala a la cabeza lineal del Hito 1 (0.996):** buscar en la galería y
  votar es tan bueno como clasificar, y además entrega ejemplos parecidos (eso es lo que agrega el Hito 3).
- **La CNN propia queda muy cerca (0.986) con la mitad de dimensiones** (256): embedding más barato de guardar.
- **Landmarks:** el vecino más cercano casi siempre acierta (R@1 0.96), pero el mAP es bajo (0.52). Hipótesis: la
  geometría centrada en la muñeca no es invariante a la rotación de la mano, así que un mismo gesto queda en
  varios grupos separados según la orientación; los vecinos inmediatos están bien, pero el ranking completo no.
- **Errores Top-1 de `resnet18_ft`** (24 de 5 500): `palm -> stop` (5), `call -> no_gesture`, `ok -> peace`,
  `peace -> one` (3 c/u). Son los mismos pares difíciles del Hito 1 (manos abiertas parecidas, dedos que se
  tapan). Con ImageNet dominan `peace <-> one` (249) y `ok -> palm` (97).

### 2.2 ¿Gesto o identidad? (`results/retrieval/identidad/`)

Dentro de test cada persona aparece ~5 veces con gestos distintos. Para cada mano se buscan las otras manos de
test **sin contar la misma foto** (misma foto = mismo fondo, sería trampa).

| | imagenet | cnn_residual | resnet18_ft |
|---|---|---|---|
| Similitud media: mismo gesto, otra persona | 0.620 | 0.933 | **0.972** |
| Similitud media: misma persona, otro gesto | 0.616 | 0.566 | 0.560 |
| Similitud media: otra persona, otro gesto (referencia) | 0.576 | 0.568 | 0.561 |
| Top-10: fracción misma persona con otro gesto (azar = 0.0012) | **0.030** (25× azar) | 0.0001 | 0.0000 |
| Consultas con la misma persona en su Top-10 (otro gesto) | **22.3 %** | 0.11 % | 0.04 % |
| Percentil medio de la misma persona entre las manos de otro gesto (50 % = azar) | **66 %** | 49.6 % | 49.4 % |

Lectura: **con `resnet18_ft` el embedding representa el gesto y no la persona.** La misma persona con otro gesto
queda igual de lejos que un desconocido (0.560 vs 0.561; percentil 49 % = azar). Con ImageNet sí se filtra
identidad: en el 22 % de las consultas aparece la misma persona haciendo otro gesto (mismo fondo, ropa, piel).
El fine tuning del Hito 1 "borró" esa información porque no ayudaba a clasificar. Ejemplos en
`identidad_ejemplos.png` (borde azul = misma persona con otro gesto).

## 3. Tareas de Tomás (Persona 2): sistema final imagen → detección → crop → embedding → Top-K

Todo lo de abajo usa piezas que **ya existen**; no hay que tocar `src/retrieval.py` ni `src/det.py`
(si hace falta algo, agregar funciones nuevas en tu archivo).

### 3.1 Retrieval con las cajas del detector (crear `src/retrieval_sistema.py`)

Pregunta: **¿cuánto se pierde cuando el recorte lo da el detector en vez de la caja perfecta?** (la parte 2.1 es
el techo con cajas GT).

Para cada imagen de test (`data/splits/det_test.json`):
1. Detector de manos del Hito 2 → cajas + score. Ya está todo en
   `results/det/det_hands_hito1/predictions_test.json` (xyxy normalizada a la imagen original). Umbral de
   operación: el elegido en val, `op_score_from_val` de `results/det/cascade/metrics_test.json`, o elegir uno
   propio en val.
2. Recortar cada caja **igual que en el Hito 1**: `from src.det_cascade import crop_hand` (cuadrado 1.3×, 128×128).
3. Embedding: `from src.retrieval import load_encoder, embed` →
   `embed(load_encoder("resnet18_ft", device), x, device)` con `x` = recortes pasados por
   `build_transforms(False, 128)`.
4. Galería: `from src.retrieval import split_embeddings` →
   `g_df, g_emb, _ = split_embeddings("resnet18_ft", "train", device)` (lee `data/embeddings/` si existe).
5. Top-K: `sims, idx = search(q_emb, g_emb, k=10, device)`. El gesto de cada vecino es `g_df.label.iloc[idx]`.
6. Para saber el gesto "real" de una caja detectada: emparejarla con la caja GT de mayor IoU (≥ 0.5). Las cajas
   sin pareja son falsos positivos y las manos GT sin detección son manos perdidas (cuentan como fallo).
7. Reportar `R@1`, `R@5`, `P@10` y kNN10 del sistema completo vs los de la tabla 2.1 (techo con GT), más
   cuántas manos se pierden en la detección. Opcional: separar por tamaño de mano (tercios < 83 / 83–117 / > 117 px,
   como en los hitos 1 y 2).

### 3.2 Demo final (crear `src/demo_retrieval.py`, o extender `src/demo.py` con una opción nueva)

`imagen → mano localizada → gesto → ejemplos parecidos + comando`. Por cada mano detectada: dibujar la caja y
al lado sus 5 vecinos de la galería; el gesto = votación de los vecinos (kNN) y el comando sale de `COMMANDS` de
`src/demo.py`. Probar con las fotos propias de `demo/Foto_Javier*.jpg` (**declararlas como imágenes externas**
en el informe) y con algunas de test (`--random`). Guardar en `results/demo_retrieval/`.

### 3.3 Figura de casos correctos/fallidos del sistema completo

Igual que `topk_ejemplos_test.png`, pero con recortes del detector: aciertos, una caja mal centrada que cambia
los vecinos, un falso positivo (¿qué recupera un recorte que no es mano?) y una mano perdida.

## 4. README (no lo toqué para no pisar tu versión)

Agregar una sección "Hito 3" con:

```bash
# Retrieval con recortes GT: galería = train, consultas = val/test
python -m src.retrieval                  # -> results/retrieval/comparison.md
python -m src.retrieval_identidad        # -> results/retrieval/identidad/resumen.md
# (tus comandos del sistema completo y la demo)
```

y agregar `data/embeddings/` al `.gitignore` (se regenera con el primer comando).

## 5. Reglas para que todo sea comparable

- Misma galería (train) y mismas consultas (test) que la tabla 2.1; **test se usa una sola vez**, umbrales en val.
- Mismo encoder (`resnet18_ft`), misma normalización (largo 1) y misma `search`.
- Semilla 42; logs en `logs/retrieval/`.
