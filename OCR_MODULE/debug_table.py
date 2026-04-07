import cv2
import numpy as np
from PIL import Image
from layout import get_predictors
import pypdfium2 as pdfium

# load your document
pdf = pdfium.PdfDocument(r"C:\Users\Dell\Downloads\Telegram Desktop\1773579070609.pdf")
page = pdf[0]
bitmap = page.render(scale=300/72)
pil_image = bitmap.to_pil()
page_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

predictors = get_predictors()

# detect layout to get table bbox
from layout import detect_layout
regions = detect_layout(pil_image)
print("Table bboxes:", regions["table"])

if regions["table"]:
    bbox = regions["table"][0]
    x1, y1, x2, y2 = bbox
    cropped_pil = pil_image.crop((x1, y1, x2, y2))

    table_results = predictors["table_rec"]([cropped_pil])
    table = table_results[0]

    print(f"\nRows: {len(table.rows)}, Cols: {len(table.cols)}, Cells: {len(table.cells)}")
    for cell in table.cells:
        print(f"  cell_id={cell.cell_id} row={cell.row_id} col={cell.col_id} rowspan={cell.rowspan} colspan={cell.colspan} merge_up={cell.merge_up} merge_down={cell.merge_down} is_header={cell.is_header}")
        print(f"    polygon={cell.polygon}")
