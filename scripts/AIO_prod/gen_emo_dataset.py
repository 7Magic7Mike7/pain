import os
import pandas as pd

VALUE_COLUMNS = \
    "anger_moral_injury,climate_fear_anxiety_panic,depression_suicidal_ideation,displacement_exile,emotional_pain_unspecified,general_fear_anxiety_panic,general_physical_pain,grief_solastalgia,helplessness_powerlessness,loss_precarity,no_pain,out_of_scope,relationship_social_pain,shame_guilt_self_blame,trauma_overwhelm,uncertainty_instability" \
    .split(",")

def _load_words(words_path: str):
  df_words = pd.read_csv(words_path)
  return {
      str(row["iso3"]): str(row["word"])
      for _, row in df_words.iterrows()
      if pd.notna(row["word"]) and str(row["word"]).strip() != ""
  }


def perform(input_path: str, words_path: str, output_path: str):
  print("Starting...")
  word_by_country = _load_words(words_path)
  df_emo = pd.read_csv(input_path)

  # 1) add a column called "aggrId" that contains pd.NA in every row
  df_emo["aggrId"] = pd.NA
  # 2) sum up all float values of the columns contained in value_columns, divide it by len(value_columns) and store it in a new column called "value"
  df_emo["value"] = df_emo[VALUE_COLUMNS].sum(axis=1) / len(VALUE_COLUMNS)
  # 3) add a column called "category" that contains "emotional" in every row (no category used for emotional data)
  df_emo["category"] = "emotional"
  # 4) add a column called "word" that contains the word associated with the iso3 code in the dictionary word_by_country
  df_emo["word"] = df_emo["iso3"].map(word_by_country)    # todo: define na_action?
  # 5) rename the column "iso3" to "country"
  df_emo = df_emo.rename(columns={"iso3": "country"})
  # 6) drop all columns named in value_columns
  df_emo = df_emo.drop(columns=[col for col in VALUE_COLUMNS if col in df_emo.columns])
  # 7) reorder columns
  column_order = ["aggrId", "value", "category", "country", "word"]
  df_emo = df_emo[column_order]
  
  # 8) Optiona: drop rows without words
  df_emo = df_emo[df_emo["word"].notna()]

  df_emo.to_csv(output_path, index=True, index_label="id")
  print("-done-")

if __name__ == "__main__":
  BASE_PATH = os.path.join("data")#'..', '..', 'data')
  DATASET_PATH = os.path.join(BASE_PATH, "dummy", "emotion-values-dummy.csv")
  WORDS_PATH = os.path.join(BASE_PATH, "dummy", "emotion-mvp-words-dummy.csv")
  OUTPUT_PATH = os.path.join(BASE_PATH, "dummy", "data_types", "emo.csv")
  perform(DATASET_PATH, WORDS_PATH, OUTPUT_PATH)
