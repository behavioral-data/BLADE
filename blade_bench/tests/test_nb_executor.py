from absl.testing import absltest
import asynctest
import pandas as pd
from blade_bench.utils import get_dataset_csv_path
from blade_bench.nb.nb_executor import NotebookExecutorBasic

FERTILITY_PATH = get_dataset_csv_path("fertility")


class TestNBExecutor(absltest.TestCase, asynctest.TestCase):

    async def test_nb_executor(self):
        try_code = "df = df.head()"
        nbexec = NotebookExecutorBasic(
            data_path=FERTILITY_PATH, save_path=".", run_init_once=True
        )
        outputs, sucess, my_var = await nbexec.run_code(try_code)

        df_head = pd.read_csv(FERTILITY_PATH).head()
        self.assertEqual(my_var.shape, df_head.shape)
        self.assertEqual(my_var.columns.tolist(), df_head.columns.tolist())
        self.assertTrue(sucess)

    async def test_nb_executor_invalid_code(self):
        invalid_code = "df = df.head(unknown_arg)"
        nbexec = NotebookExecutorBasic(
            data_path=FERTILITY_PATH, save_path=".", run_init_once=True
        )
        outputs, success, my_var = await nbexec.run_code(invalid_code)

        self.assertFalse(success)
        self.assertIsNone(my_var)


if __name__ == "__main__":
    absltest.main()
