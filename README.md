# verificador-de-veracidad-enfoque-eclesial
Verificador de veracidad con una base cerrada de documentos de la iglesia catololica, contrasta otros documentos(noticias o articulos) frente a la base de verdad proporcionada, haciendo uso de un scrapper web y un IVR que alimenta un modelo random forest para decidir la alineación del material contrastado

#### Importante
Si se quiere agregar un .env para que sea leible se debe adaptar la clase `Settings` de config.py (preguntar a la IA)
Si se instala una nueva libreria se debe agregar en `pyproject.toml`
Para instalar las librerias se debe hacer ``pip install .`` en la carpeta raiz y con el .venv activado

### Configuracion basica
Las rutas, variables estaticas, etc; deben ir en `config.toml`, si se crea una nueva seccion se debe actualizar el archivo de configuracion `config.py`
Todas las variables se leen con pydantic 

#### Ejemplo para leer las variables:
~~~ python
from config import obtener_configuraciones 

settings = obtener_configuraciones()
print(settings.test.base_datos)
~~~


#### Ejemplo para agregar:
Archivo `config.toml`
~~~ 
...
[test]
base_datos = "/test/db/"
batch_size = 200
...
~~~

Archivo `config.py`
~~~ python
...
class TestConfig(BaseModel):
    base_datos: str
    batch_size: int
...

...
class Settings(BaseSettings):
    test: TestConfig
...
~~~


### Ejecutar el Extractor del Corpus (ETL)

Las configuraciones de este módulo se controlan desde el archivo `config.toml` (en la sección `[corp_extractor]`). Dependiendo de la variable `modo_ejecucion`, el pipeline orquestará el proceso de la siguiente manera:

* **Modo `"dinamico"`:** Se conecta a internet (`vatican.va`), **descarga** en tiempo real todos los textos oficiales, guarda los archivos crudos y luego los pasa por el **clasificador** para inyectarles la categoría y el nivel de autoridad.
* **Modo `"estatico"`:** No usa internet. Lee los archivos JSON crudos que ya están descargados localmente y los pasa directamente por el **clasificador**.

Para correr todo el proceso centralizado, ejecuta el siguiente comando desde la **raíz del proyecto** (con el entorno virtual activado):

~~~ bash
python src/corp_extractor/main_etl.py
~~~ 
