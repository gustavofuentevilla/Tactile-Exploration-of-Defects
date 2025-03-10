# Tactile-Exploration-of-Defects

En la carpeta 1, se muestran cálculos previos e importantes para la implementación de Ergodic Control en dos versiones: un archivo Live Script con detalle de ecuaciones, y un Script normal sólo con código y comentarios.

En 2, se implementa el Ergodic Control como un problema de optimización con ayuda de CasAdi, usando un sistema de primer orden, el archivo principal es ErgCtrl_1OrderSys_OptProblem. 

En 3, se implementa el Ergodic Control para un sistema de segundo orden, el archivo principal es ErgCtrl_2OrdSys_OptProblem.

En 4, se usa el Ergodic Control para la tarea de exploración de defectos, considerando sólo un defecto y un estimador de la función de densidad de probabilidad (PDF) basado en la regla de Bayes y un modelo de medición de un sensor táctil, dada una PDF real desconocida. El archivo principal es ErgCtrl_Exploration_SingleDef.

En 5, se extiende el problema de exploración de defectos a múltiples defectos; sin embargo, este caso no es realista. Se consideran modelos de mediciones táctiles independientes para cada defecto. Esto no es realista porque es como tener tantos sensores abordo de un robot como defectos hay, y que cada sensor sólo arroje mediciones respecto de su defecto (y no arroje nada cuando pasa sobre los demás defectos). El archivo principal es ErgCtrl_Exploration_MultiDef. El número de defectos es conocido.

En 6, se resuelve el problema identificando los posibles defectos agrupando las mediciones en el espacio con la técnica Gaussian Mixture Model (GMM), los modelos independientes se formulan a partir de esos grupos, y la localización de defectos se estima con el mismo estimador usado en 4 y 5, tratando a los defectos de forma independiente. Esta solución ya es realista porque los modelos se formulan a partir del resultado de GMM (no simplemente se definen como en 5). El archivo principal es ErgCtrl_Exploration_MultiDef. El número de defectos es conocido.

Nota: En cada carpeta se necesita un archivo M.casadi para correr el código principal, ese archivo se puede generar con el código comentado que se encuentra en el mismo código principal. También se puede usar el código Casadi_form de la carpeta Casadi_Formulation_ExplTask para generar el archivo M.casadi en cualquier caso.
