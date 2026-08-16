from typing import List
import argparse
import numpy as np
import os
import pandas as pd


DEFAULT_BASE_PATH = os.path.join("data")#'..', '..', 'data')
DEFAULT_INPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "dummy", "emotion-values-dummy.csv")
DEFAULT_WORDS_PATH = os.path.join(DEFAULT_BASE_PATH, "dummy", "emotion-mvp-words-dummy.csv")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_BASE_PATH, "dummy", "data_types", "emo.csv")

VALUE_COLUMNS = \
    "anger_moral_injury,climate_fear_anxiety_panic,depression_suicidal_ideation,displacement_exile,emotional_pain_unspecified,general_fear_anxiety_panic,general_physical_pain,grief_solastalgia,helplessness_powerlessness,loss_precarity,no_pain,out_of_scope,relationship_social_pain,shame_guilt_self_blame,trauma_overwhelm,uncertainty_instability" \
    .split(",")

def _init_value_columns() -> List[str]:
  # deep copy VALUE_COLUMNS
  value_columns = [vc for vc in VALUE_COLUMNS]
  # ignore no_pain & out_of_scope as they do not represent emotional pain
  value_columns.remove("no_pain")
  value_columns.remove("out_of_scope")
  return value_columns

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
  value_columns = _init_value_columns()
  df_emo = pd.read_csv(input_path)

  # 1) add a column called "aggrId" that contains pd.NA in every row
  df_emo["aggrId"] = pd.NA
  # 2) add a column called "category" that contains "emotional" in every row (no category used for emotional data)
  df_emo["category"] = "emotional"
  # 3) sum up all float values of the columns contained in value_columns, divide it by len(value_columns) and store it in a new column called "value"
  df_emo["value"] = df_emo[value_columns].sum(axis=1) / len(value_columns)
  # 4) add a column called "word" that contains the word associated with the iso3 code in the dictionary word_by_country
  df_emo["word"] = df_emo["iso3"].map(word_by_country)    # todo: define na_action?
  # 5) rename the column "iso3" to "country"
  df_emo = df_emo.rename(columns={"iso3": "country"})
  # 6) drop all columns named in value_columns
  df_emo = df_emo.drop(columns=[col for col in value_columns if col in df_emo.columns])
  
  # 7) Optional: drop rows without words
  df_emo = df_emo[df_emo["word"].notna()]

  # 8) compute id column
  df_emo["id"] = np.arange(1, len(df_emo) + 1)
  # 9) reorder columns
  column_order = ["id", "aggrId", "value", "category", "country", "word"]
  df_emo = df_emo[column_order]

  df_emo.to_csv(output_path, index=False)
  print("-done-")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog='Emotional Pain Dataset Generation Script',
    description='Generates the emotional pain dataset from a sentiment analysis dataset.'
  )
  parser.add_argument("-bp", "--base-path", type=str, help="common base path shared among input and output file")
  parser.add_argument("-i", "-in", "--input", type=str, help="path to the input file")
  parser.add_argument("-o", "-out", "--output", type=str, help="path to the output file")
  parser.add_argument("-w", "--word-data", type=str, help="path to the word data (input) file")

  args = parser.parse_args()
  # parse paths
  if args.base_path:
    if args.input:
      input_path = os.path.join(args.base_path, args.input)
    else:
      print("Specifying a base_path requires an input argument but none was provided!")
      exit(1)
    if args.output:
      output_path = os.path.join(args.base_path, args.output)
    else:
      print("Specifying a base_path requires an output argument but none was provided!")
      exit(1)
    if args.word_data:
      words_path = os.path.join(args.base_path, args.word_data)
    else:
      print("Specifying a base_path requires a word-data argument but none was provided!")
      exit(1)
  else:
    input_path = args.input if args.input else DEFAULT_INPUT_PATH
    output_path = args.output if args.output else DEFAULT_OUTPUT_PATH
    words_path = args.word_data if args.word_data else DEFAULT_WORDS_PATH
    
  perform(input_path, words_path, output_path)
