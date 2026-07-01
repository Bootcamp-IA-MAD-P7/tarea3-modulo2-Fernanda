# Parte 1: Algoritmos de Clasificación en Machine Learning Supervisado

## 1. ¿Qué es la clasificación en Machine Learning y cuál es su propósito?

La **clasificación** es una técnica de **Machine Learning supervisado** que permite entrenar un modelo para que aprenda a asignar una **categoría** o **clase** a partir de datos de ejemplo.

Se llama **aprendizaje supervisado** porque el modelo aprende usando datos que ya tienen una respuesta correcta. Es decir, durante el entrenamiento se le muestran ejemplos donde ya sabemos cuál era la clase real.

Por ejemplo, si queremos predecir si un estudiante aprobará o no un examen, podemos entrenar el modelo con datos históricos:

| Horas de estudio | Asistencia | Resultado real |
|---:|---|---|
| 8 | Alta | Aprueba |
| 2 | Baja | No aprueba |
| 6 | Media | Aprueba |
| 1 | Baja | No aprueba |

El modelo analiza esos ejemplos y aprende patrones. Después, cuando recibe un nuevo caso, intenta predecir a qué clase pertenece.

La clasificación no busca predecir un número exacto, sino una **etiqueta**. Por eso, no responde preguntas como “¿qué nota sacará el estudiante?”, sino preguntas como “¿aprueba o no aprueba?”.

El propósito de la clasificación es ayudar a tomar decisiones cuando la respuesta posible pertenece a un conjunto limitado de opciones.

Algunos ejemplos de clasificación en el mundo real son:

| Problema | Datos de entrada | Clase que se quiere predecir |
|---|---|---|
| Detectar spam | Texto de un email | Spam / No spam |
| Predecir aprobación de estudiantes | Horas de estudio, asistencia, participación | Aprueba / No aprueba |
| Detectar fraude bancario | Monto, país, horario, historial del cliente | Fraude / No fraude |
| Diagnóstico médico inicial | Síntomas, edad, resultados de análisis | Enfermo / Sano |
| Clasificar opiniones | Texto de una reseña | Positiva / Negativa / Neutral |

Un ejemplo práctico sería un banco que quiere detectar operaciones fraudulentas. El modelo puede analizar información como el importe de la compra, el país donde se realizó, el horario, el dispositivo usado y el historial del cliente. Con esos datos, puede clasificar la operación como “fraudulenta” o “no fraudulenta”.

---

## 2. Diferencia entre clasificación binaria y clasificación multiclase

La clasificación puede tener diferentes tipos de salida. Las más comunes son la **clasificación binaria** y la **clasificación multiclase**.

### Clasificación binaria

La **clasificación binaria** se usa cuando solo hay dos posibles clases.

Ejemplos:

| Problema | Clases posibles |
|---|---|
| Detectar spam | Spam / No spam |
| Predecir aprobación | Aprueba / No aprueba |
| Detectar fraude | Fraude / No fraude |
| Predecir abandono de clientes | Abandona / No abandona |
| Diagnóstico simple | Enfermo / Sano |

Se usa clasificación binaria cuando la decisión final es de tipo **sí/no**, **verdadero/falso** o **positivo/negativo**.

Por ejemplo, si queremos saber si una operación bancaria es fraudulenta o no, estamos ante un problema binario porque solo existen dos respuestas posibles:

- Fraude
- No fraude

Aunque el modelo analice muchas variables para tomar la decisión, sigue siendo clasificación binaria si la salida final tiene solo dos clases.

### Clasificación multiclase

La **clasificación multiclase** se usa cuando hay más de dos clases posibles.

Ejemplos:

| Problema | Clases posibles |
|---|---|
| Clasificar tipo de flor | Setosa / Versicolor / Virginica |
| Clasificar nivel de riesgo | Bajo / Medio / Alto |
| Clasificar tickets de soporte | Técnico / Facturación / Comercial / Legal |
| Clasificar noticias | Política / Deportes / Economía / Tecnología |
| Clasificar imágenes de animales | Perro / Gato / Caballo / Pájaro |

Por ejemplo, si en lugar de predecir solamente “fraude” o “no fraude”, queremos saber qué tipo de fraude ocurrió, podríamos tener varias clases:

- No fraude
- Fraude por robo de tarjeta
- Fraude por identidad falsa
- Fraude por cuenta tomada
- Fraude por triangulación

En ese caso, el problema sería multiclase porque el modelo debe elegir una opción entre varias categorías posibles.

La diferencia principal es que en clasificación binaria el modelo decide entre dos clases, mientras que en clasificación multiclase decide entre tres o más clases.

---

## 3. Principales algoritmos de clasificación supervisada

Los algoritmos de clasificación son métodos que permiten que un modelo aprenda patrones a partir de datos etiquetados. Cada algoritmo tiene una forma distinta de aprender y de tomar decisiones.

### 3.1 Regresión Logística

La **regresión logística** es uno de los algoritmos más usados para problemas de clasificación binaria.

Aunque su nombre incluye la palabra “regresión”, se utiliza para clasificar. Su objetivo es calcular la **probabilidad** de que un caso pertenezca a una clase.

Por ejemplo, en un problema donde queremos predecir si un estudiante aprueba, el modelo podría devolver esto:

| Estudiante | Horas de estudio | Probabilidad de aprobar | Predicción |
|---|---:|---:|---|
| A | 8 | 0.92 | Aprueba |
| B | 5 | 0.65 | Aprueba |
| C | 2 | 0.20 | No aprueba |

Normalmente se define un umbral. Si la probabilidad es igual o mayor a 0.5, el modelo predice la clase positiva. Si es menor a 0.5, predice la clase negativa.

Ejemplo:

| Probabilidad estimada | Predicción |
|---:|---|
| 0.80 | Aprueba |
| 0.55 | Aprueba |
| 0.40 | No aprueba |
| 0.10 | No aprueba |

Es útil cuando necesitamos un modelo simple, rápido e interpretable. Se usa mucho en problemas como:

- Predecir si un estudiante aprueba o no.
- Detectar si un email es spam o no.
- Predecir si un cliente abandonará un servicio.
- Estimar si una operación puede ser fraudulenta.

### 3.2 Árboles de Decisión

Los **árboles de decisión** funcionan como una secuencia de preguntas. Cada pregunta divide los datos hasta llegar a una predicción final.

Ejemplo simple para predecir si un estudiante aprueba:

```text
¿Estudió más de 5 horas?
│
├── Sí → ¿Asistió a clase?
│       ├── Sí → Aprueba
│       └── No → Puede no aprobar
│
└── No → No aprueba
```

El modelo aprende qué preguntas conviene hacer y en qué orden. Busca dividir los datos de manera que las clases queden lo más separadas posible.

Ejemplos de preguntas que podría hacer un árbol:

| Variable | Pregunta posible |
|---|---|
| Horas de estudio | ¿Estudió más de 5 horas? |
| Edad | ¿Tiene más de 30 años? |
| Ingresos | ¿Gana más de cierta cantidad? |
| Historial de pagos | ¿Tuvo impagos previos? |
| Asistencia | ¿Asistió a más del 80% de las clases? |

Una ventaja importante de los árboles de decisión es que son fáciles de interpretar. Se parecen a una toma de decisiones humana.

Sin embargo, si el árbol crece demasiado, puede memorizar los datos de entrenamiento y funcionar peor con datos nuevos. A eso se le llama **overfitting** o sobreajuste.

### 3.3 Support Vector Machine, SVM

**Support Vector Machine**, o **SVM**, es un algoritmo que intenta separar las clases mediante una frontera de decisión.

Imaginemos que tenemos datos de estudiantes representados en un gráfico según sus horas de estudio y asistencia. Algunos aprobaron y otros no aprobaron. SVM intenta encontrar la mejor línea o frontera para separar ambos grupos.

La idea no es solo separar las clases, sino hacerlo dejando el mayor margen posible entre ellas.

Ejemplo simplificado:

```text
No aprueba        |        Aprueba
❌ ❌ ❌ ❌        |        ✅ ✅ ✅ ✅
```

La línea vertical representa la frontera que separa las dos clases.

SVM puede funcionar bien cuando las clases están bastante separadas. También puede trabajar con problemas más complejos usando técnicas que permiten crear fronteras no lineales.

Se usa en problemas como:

- Clasificación de imágenes.
- Clasificación de textos.
- Detección de spam.
- Clasificación médica.

Una desventaja es que puede ser más difícil de interpretar que una regresión logística o un árbol de decisión.

### 3.4 K-Nearest Neighbors, KNN

**K-Nearest Neighbors**, o **KNN**, significa “K vecinos más cercanos”.

Este algoritmo clasifica un nuevo dato comparándolo con los datos más parecidos que ya conoce. La letra **K** indica cuántos vecinos cercanos se tendrán en cuenta para tomar la decisión.

Por ejemplo, si K = 5, el modelo mira los 5 casos más cercanos al nuevo dato. Luego observa cuál es la clase más común entre esos vecinos.

Ejemplo:

| Estudiante parecido | Horas de estudio | Resultado real |
|---|---:|---|
| A | 5.1 | Aprueba |
| B | 4.8 | Aprueba |
| C | 5.3 | Aprueba |
| D | 3.5 | No aprueba |
| E | 5.0 | Aprueba |

Si llega un nuevo estudiante que estudió aproximadamente 5 horas, KNN buscará estudiantes parecidos. Como la mayoría de los vecinos aprobaron, el modelo puede predecir que el nuevo estudiante también aprobará.

Otro ejemplo: si tenemos datos de niños, edad, nivel de actividad y juegos favoritos, KNN podría predecir el juego favorito de un nuevo niño comparándolo con otros niños similares.

KNN es intuitivo porque se basa en la idea de similitud: un dato nuevo probablemente pertenezca a la misma clase que los datos más parecidos a él.

Una desventaja es que puede ser lento con muchos datos, porque necesita comparar el nuevo caso con muchos ejemplos previos.

### 3.5 Naive Bayes

**Naive Bayes** es un algoritmo basado en probabilidades. Calcula qué clase es más probable según las características del dato.

Se usa mucho en clasificación de texto, especialmente para detectar spam.

Ejemplo:

Un email contiene palabras como:

```text
gratis, premio, urgente, gana, oferta
```

Si esas palabras aparecen frecuentemente en emails marcados como spam, el modelo puede calcular que ese email tiene alta probabilidad de ser spam.

La lógica sería:

| Palabras del email | Clase más probable |
|---|---|
| gratis, premio, urgente | Spam |
| reunión, informe, equipo | No spam |
| factura, pago, vencimiento | Facturación |

Se llama “naive”, que significa “ingenuo”, porque parte de una suposición simplificada: considera que las variables son independientes entre sí. En la realidad, esto no siempre sucede, pero el algoritmo puede funcionar bien en muchos casos.

Es útil para:

- Detectar spam.
- Clasificar noticias.
- Analizar sentimientos en comentarios.
- Clasificar tickets de soporte.

---

## 4. Métricas para evaluar modelos de clasificación

Una vez que entrenamos un modelo, necesitamos evaluar si está funcionando bien. Para eso usamos métricas.

No alcanza con decir que un modelo “acierta mucho”. También necesitamos saber qué tipo de errores comete, porque no todos los errores tienen el mismo impacto.

Por ejemplo, no es igual equivocarse recomendando una película que equivocarse al detectar una enfermedad grave o una operación fraudulenta.

### 4.1 Matriz de Confusión

La **matriz de confusión** permite comparar las predicciones del modelo con los valores reales.

En un problema binario, por ejemplo “aprueba” o “no aprueba”, pueden ocurrir cuatro situaciones:

| Caso | Significado |
|---|---|
| Verdadero Positivo, TP | El modelo predijo “aprueba” y realmente aprobó |
| Verdadero Negativo, TN | El modelo predijo “no aprueba” y realmente no aprobó |
| Falso Positivo, FP | El modelo predijo “aprueba”, pero realmente no aprobó |
| Falso Negativo, FN | El modelo predijo “no aprueba”, pero realmente sí aprobó |

La matriz puede verse así:

|  | Real: Aprueba | Real: No aprueba |
|---|---:|---:|
| Predice: Aprueba | Verdadero Positivo | Falso Positivo |
| Predice: No aprueba | Falso Negativo | Verdadero Negativo |

La matriz de confusión es importante porque muestra no solo cuántas veces acertó el modelo, sino también en qué se equivocó.

### 4.2 Accuracy, o exactitud

La **accuracy** mide el porcentaje total de predicciones correctas.

Fórmula:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Ejemplo:

Si un modelo hizo 100 predicciones y acertó 85, entonces:

```text
Accuracy = 85 / 100 = 0.85
```

Eso significa que tiene un 85% de exactitud.

La accuracy es útil cuando las clases están equilibradas.

Ejemplo equilibrado:

| Clase | Cantidad de casos |
|---|---:|
| Aprueba | 50 |
| No aprueba | 50 |

Pero puede ser engañosa cuando las clases están desbalanceadas.

Ejemplo de fraude:

| Clase | Cantidad de casos |
|---|---:|
| No fraude | 990 |
| Fraude | 10 |

Si un modelo predice siempre “no fraude”, acertaría 990 de 1000 casos. Tendría 99% de accuracy, pero sería un mal modelo porque no detectaría ningún fraude.

Por eso, en problemas desbalanceados conviene mirar otras métricas además de accuracy.

### 4.3 Precision, o precisión

La **precision** mide qué tan confiables son las predicciones positivas del modelo.

Responde esta pregunta:

> De todo lo que el modelo predijo como positivo, ¿cuánto era realmente positivo?

Fórmula:

```text
Precision = TP / (TP + FP)
```

Ejemplo con fraude:

Si el modelo marcó 20 operaciones como fraude, pero solo 15 eran realmente fraude:

```text
Precision = 15 / 20 = 0.75
```

La precisión es del 75%.

Esto significa que cuando el modelo dice “esto es fraude”, acierta el 75% de las veces.

Precision es especialmente importante cuando queremos evitar falsos positivos.

Ejemplos:

| Problema | Por qué importa precision |
|---|---|
| Fraude bancario | No queremos bloquear muchas operaciones legítimas |
| Spam | No queremos enviar emails importantes a spam |
| Selección de candidatos | No queremos descartar personas válidas por error |

### 4.4 Recall, o sensibilidad

El **recall** mide cuántos casos positivos reales logró detectar el modelo.

Responde esta pregunta:

> De todos los casos que realmente eran positivos, ¿cuántos encontró el modelo?

Fórmula:

```text
Recall = TP / (TP + FN)
```

Ejemplo médico:

Si había 100 personas enfermas y el modelo detectó correctamente a 90:

```text
Recall = 90 / 100 = 0.90
```

El recall es del 90%.

Recall es especialmente importante cuando no queremos que se escapen casos positivos.

Ejemplos:

| Problema | Por qué importa recall |
|---|---|
| Diagnóstico médico | No queremos clasificar como sana a una persona enferma |
| Fraude grave | No queremos dejar pasar operaciones fraudulentas |
| Abandono de clientes | No queremos perder clientes sin detectarlo a tiempo |

En un problema médico grave, suele importar mucho el recall porque un falso negativo puede ser peligroso: el modelo dice que la persona no está enferma, pero en realidad sí lo está.

### 4.5 F1-Score

El **F1-Score** combina precision y recall en una sola métrica.

Sirve cuando queremos buscar un equilibrio entre ambas.

Fórmula:

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

No es necesario memorizar la fórmula al principio. Lo importante es entender que F1 ayuda a evaluar si el modelo mantiene un buen balance entre:

- No marcar demasiados falsos positivos.
- No dejar escapar demasiados positivos reales.

Ejemplo:

| Modelo | Precision | Recall | Interpretación |
|---|---:|---:|---|
| A | 0.95 | 0.40 | Se equivoca poco cuando predice positivo, pero se le escapan muchos positivos |
| B | 0.60 | 0.95 | Encuentra muchos positivos, pero también marca muchos falsos positivos |
| C | 0.82 | 0.80 | Tiene un comportamiento más equilibrado |

F1-Score suele ser útil cuando las clases están desbalanceadas o cuando precision y recall son importantes al mismo tiempo.

### 4.6 ROC-AUC

**ROC-AUC** mide qué tan bien el modelo puede separar una clase de otra.

Muchos modelos no solo predicen una clase, sino también una probabilidad.

Ejemplo:

| Estudiante | Probabilidad de aprobar |
|---|---:|
| A | 0.92 |
| B | 0.75 |
| C | 0.48 |
| D | 0.20 |

Para convertir esa probabilidad en una clase, se usa un umbral. Si el umbral es 0.5:

| Probabilidad | Predicción |
|---:|---|
| Mayor o igual a 0.5 | Aprueba |
| Menor a 0.5 | No aprueba |

La curva ROC evalúa cómo cambia el comportamiento del modelo cuando se prueban diferentes umbrales.

AUC significa “área bajo la curva”. Su valor suele estar entre 0 y 1.

Interpretación simplificada:

| ROC-AUC | Interpretación |
|---:|---|
| 0.5 | El modelo no distingue mejor que el azar |
| 0.7 | Separación aceptable |
| 0.8 | Buena separación |
| 0.9 o más | Muy buena separación |
| 1.0 | Separación perfecta |

ROC-AUC es útil para saber si el modelo distingue bien entre clases, más allá de un único umbral de decisión.

### Resumen de métricas

| Métrica | Pregunta que responde |
|---|---|
| Accuracy | ¿Cuántas predicciones acertó en total? |
| Precision | Cuando dijo “positivo”, ¿cuántas veces acertó? |
| Recall | De todos los positivos reales, ¿cuántos encontró? |
| F1-Score | ¿Qué tan bien equilibra precision y recall? |
| Matriz de Confusión | ¿Qué tipos de aciertos y errores tuvo? |
| ROC-AUC | ¿Qué tan bien separa una clase de otra? |

---

## 5. Feature engineering y feature selection en clasificación

En Machine Learning, las variables que usamos para entrenar un modelo suelen llamarse **features** o características.

Por ejemplo, si queremos predecir si un estudiante aprueba, las features podrían ser:

- Horas de estudio.
- Asistencia.
- Participación en clase.
- Entregas realizadas.
- Nota de trabajos prácticos.
- Uso de plataforma educativa.

### Feature engineering

El **feature engineering** consiste en crear, transformar o mejorar variables para que el modelo pueda aprender mejor.

No siempre los datos originales vienen en la forma más útil. A veces hay que prepararlos.

Ejemplo simple:

Supongamos que tenemos estas variables:

| Estudiante | Horas de estudio | Horas de sueño | Asistencia |
|---|---:|---:|---|
| A | 8 | 7 | 90% |
| B | 2 | 5 | 40% |

Podemos crear nuevas variables que ayuden al modelo:

| Nueva feature | Cómo se construye | Por qué puede servir |
|---|---|---|
| Estudia mucho | Si estudia más de 5 horas | Puede separar mejor aprobados y no aprobados |
| Buena asistencia | Si asiste a más del 80% | Puede relacionarse con aprobar |
| Riesgo académico | Pocas horas de estudio + baja asistencia | Puede detectar estudiantes con mayor riesgo |

Otros ejemplos de feature engineering:

| Situación | Transformación posible |
|---|---|
| Fecha de compra | Extraer día de la semana, mes u horario |
| Texto de email | Contar palabras sospechosas o transformar texto en vectores |
| Edad | Crear grupos: menor de 18, adulto, mayor de 65 |
| Ingresos | Crear ratio deuda/ingresos |
| Dirección | Extraer ciudad o zona |

El objetivo del feature engineering es darle al modelo información más clara y útil.

### Feature selection

El **feature selection** consiste en elegir cuáles variables usar y cuáles descartar.

No siempre más variables significan mejor modelo. Algunas variables pueden ser irrelevantes, repetidas o incluso perjudiciales.

Ejemplo:

Para predecir si un estudiante aprueba, estas variables podrían ser útiles:

- Horas de estudio.
- Asistencia.
- Entregas realizadas.

Pero estas podrían no aportar mucho:

- Color favorito.
- Número de calzado.
- Primera letra del nombre.

Feature selection ayuda a quedarse con las variables que realmente aportan valor.

Ventajas de seleccionar bien las features:

- El modelo puede ser más simple.
- Puede entrenar más rápido.
- Puede evitar ruido innecesario.
- Puede mejorar el rendimiento.
- Puede reducir el riesgo de overfitting.

Diferencia principal:

| Concepto | Qué hace |
|---|---|
| Feature engineering | Crea o transforma variables |
| Feature selection | Elige qué variables se quedan en el modelo |

---

## 7. Hiperparámetros y ajuste del modelo

Los **hiperparámetros** son configuraciones que se definen antes de entrenar un modelo.

No son valores que el modelo aprende solo a partir de los datos, sino decisiones que toma la persona que entrena el modelo.

Ejemplos de hiperparámetros:

| Algoritmo | Hiperparámetro | Qué controla |
|---|---|---|
| KNN | K | Cuántos vecinos se tienen en cuenta |
| Árbol de Decisión | max_depth | Profundidad máxima del árbol |
| SVM | C | Nivel de penalización por errores |
| Regresión Logística | C | Regularización del modelo |
| Random Forest | n_estimators | Cantidad de árboles |

Ejemplo con KNN:

Si K = 1, el modelo mira solo el vecino más cercano. Puede ser muy sensible al ruido.

Si K = 15, el modelo mira más vecinos. Puede ser más estable, pero también menos específico.

Por eso, elegir hiperparámetros adecuados puede mejorar mucho el rendimiento de un modelo.

### Hyperparameter tuning

El **hyperparameter tuning** es el proceso de probar diferentes combinaciones de hiperparámetros para encontrar las que funcionan mejor.

No se trata de elegir valores al azar sin criterio, sino de comparar resultados usando datos de validación o validación cruzada.

Dos métodos comunes son:

- Grid Search.
- Random Search.

### Grid Search

**Grid Search** prueba todas las combinaciones posibles dentro de una lista de valores definida previamente.

Ejemplo con KNN:

Queremos probar estos valores:

```text
K = 3, 5, 7
weights = uniform, distance
```

Grid Search probaría todas las combinaciones:

| Combinación | K | weights |
|---|---:|---|
| 1 | 3 | uniform |
| 2 | 3 | distance |
| 3 | 5 | uniform |
| 4 | 5 | distance |
| 5 | 7 | uniform |
| 6 | 7 | distance |

Después compara los resultados y selecciona la mejor combinación.

Ventaja:

- Es ordenado y exhaustivo.

Desventaja:

- Puede ser lento si hay muchas combinaciones.

### Random Search

**Random Search** no prueba todas las combinaciones. En lugar de eso, selecciona combinaciones aleatorias dentro de un rango de valores.

Ejemplo:

En lugar de probar todos los valores posibles de K entre 1 y 100, Random Search puede probar solo algunos valores al azar, como:

```text
K = 4, 11, 27, 52, 89
```

Ventaja:

- Puede ser más rápido que Grid Search.
- Es útil cuando hay muchos hiperparámetros o rangos muy amplios.

Desventaja:

- Puede no probar justo la mejor combinación posible.

### Diferencia entre Grid Search y Random Search

| Método | Cómo busca | Ventaja principal | Desventaja principal |
|---|---|---|---|
| Grid Search | Prueba todas las combinaciones definidas | Es completo y ordenado | Puede ser lento |
| Random Search | Prueba combinaciones aleatorias | Es más rápido en espacios grandes | Puede saltarse combinaciones buenas |

En la práctica, si tenemos pocos hiperparámetros y pocas combinaciones, Grid Search puede ser suficiente. Si tenemos muchos valores posibles, Random Search puede ser más eficiente.

---

## Conexión con la práctica de regresión logística

En la parte práctica, el problema de predecir si un estudiante aprobará según sus horas de estudio es un caso de **clasificación binaria**. La variable de entrada principal será la cantidad de horas de estudio y la salida esperada será una clase: **aprueba** o **no aprueba**.

La regresión logística es adecuada para este ejercicio porque permite calcular una probabilidad de aprobación y luego convertir esa probabilidad en una predicción usando un umbral, por ejemplo 0.5.

Ejemplo:

| Horas de estudio | Probabilidad estimada de aprobar | Predicción final |
|---:|---:|---|
| 1 | 0.12 | No aprueba |
| 3 | 0.42 | No aprueba |
| 5 | 0.68 | Aprueba |
| 8 | 0.94 | Aprueba |

Para evaluar el modelo, no solo se puede mirar cuántas predicciones acertó. También conviene revisar la matriz de confusión, precision, recall y F1-Score para entender mejor qué tipo de errores cometió.

---

## Parte 2: práctica guiada de regresión logística

El ejercicio práctico está desarrollado en el notebook [`parte2_regresion_logistica.ipynb`](parte2_regresion_logistica.ipynb).

La práctica usa un dataset sintético y reproducible para predecir si un estudiante aprueba según sus horas de estudio. Incluye exploración de datos, visualizaciones, entrenamiento con `LogisticRegression`, métricas de clasificación, ROC-AUC, curva sigmoide y ejemplos de predicción con nuevos estudiantes.

Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

Para abrir el notebook:

```bash
jupyter notebook parte2_regresion_logistica.ipynb
```
