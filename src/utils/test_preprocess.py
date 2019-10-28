import pytest
import pandas as pd
import preprocess

ORDINALITY_MAPPING = {"third": {"d": 1, "a": 2}}


@pytest.fixture
def test_df():
    return pd.DataFrame(
        data={
            "first": ["a", "b", "c", "d", "e"],
            "second": ["a", "a", "b", "b", "d"],
            "third": ["d", "d", "a", "a", "d"],
        }
    )


def test_autoencode_ordinality_mapping(test_df):
    encod = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )
    encod.fit(test_df)
    res = encod.transform(test_df)
    assert list(res["third"]) == [1, 1, 2, 2, 1]


def test_autoencode_count_encoding(test_df):
    encod = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )
    encod.fit(test_df)
    res = encod.transform(test_df)
    assert list(res["first"]) == [1, 1, 1, 1, 1]


def test_autoencode_one_hot_encoding(test_df):
    encod = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )
    encod.fit(test_df)
    res = encod.transform(test_df)
    assert list(res["second_b"]) == [0, 0, 1, 1, 0]


def test_autoencode_fit_and_transform(test_df):
    encod = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )
    encod.fit(test_df)
    res = encod.transform(test_df)

    assert list(sorted(res.columns)) == [
        "first",
        "len_first",
        "second_a",
        "second_b",
        "second_d",
        "third",
    ]


def test_autoencode_fit_transform(test_df):
    encod = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )
    encod_2 = preprocess.AutoEncoder(
        cardinality_threshold=5, ordinality_mapping=ORDINALITY_MAPPING
    )

    encod.fit(test_df)
    res = encod.transform(test_df)

    res_2 = encod_2.fit_transform(test_df)

    assert list(sorted(res_2.columns)) == list(sorted(res.columns))


def test_column_dropper(test_df):
    col_remover = preprocess.ColumnRemover(drop_columns=["second"])
    col_remover.fit(test_df)
    res = col_remover.transform(test_df)

    assert list(sorted(res.columns)) == ["first", "third"]
