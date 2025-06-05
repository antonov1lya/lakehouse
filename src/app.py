import os

from elt import ELTPipeline
from ml import MLPipeline
from session import Session

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'


def main():
    session = Session()
    spark = session.get_session()

    elt_pipeline = ELTPipeline(spark)
    elt_pipeline.transform()

    ml_pipeline = MLPipeline(spark)
    ml_pipeline.run()


if __name__ == "__main__":
    main()
