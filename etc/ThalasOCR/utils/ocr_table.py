import time
import pandas as pd
from .utils import *

def ocr_table(paddleocr, vietocr, image, image_name):
    """[summary]

    Args:
        image ([type]): [description]

    Raises:
        Exception: [description]
    """
    start_time = time.time()

    # best size for model detect text
    image = resize_image(image)

    # detect and ocr all text in talbe
    box_texts = paddleocr.ocr(image, rec=False)
    result = format_ocr_result(vietocr, image, box_texts)

    df = pd.DataFrame(result, columns=["TenXetNghiem", "KetQua"])

    # get from row "Ten Xet Nghiem" 
    first_row = 0
    for index, row in df.iterrows():
        if similar(row["TenXetNghiem"], "TÊN XÉT NGHIỆM"):
            first_row = index
            break

    df = df.iloc[first_row:] 
    result = df.to_json(orient="records")

    print(df)
    # save to excel 
    # writer = pd.ExcelWriter(f"./excel_output/{image_name[:-4]}.xlsx")
    # df.to_excel(writer, sheet_name = "ThalasOCR", header = False, index = False)
    # writer.save()

    print("Time inference: ", time.time() - start_time)
    return result