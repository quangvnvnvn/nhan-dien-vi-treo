/** Tạo workbook Excel hằng ngày bằng @oai/artifact-tool. */
import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function text(value) {
  const normalized = String(value ?? "--");
  return normalized.startsWith("=") ? `'${normalized}` : normalized;
}

function number(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

async function readPayload() {
  let input = "";
  for await (const part of process.stdin) input += part;
  const payload = JSON.parse(input);
  if (!payload || typeof payload.output_path !== "string" || !Array.isArray(payload.records)) {
    throw new Error("Dữ liệu xuất Excel không hợp lệ.");
  }
  return payload;
}

async function buildWorkbook(payload) {
  const records = payload.records;
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Kết quả");
  sheet.showGridLines = false;

  sheet.getRange("A1:I1").merge();
  sheet.getRange("A1").values = [[`BÁO CÁO NHẬN DIỆN VỈ - ${text(payload.date)}`]];
  sheet.getRange("A1:I1").format = {
    fill: "#0F172A",
    font: { bold: true, color: "#FFFFFF", size: 15 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange("A1:I1").format.rowHeight = 28;
  sheet.getRange("A2:I2").merge();
  sheet.getRange("A2").values = [[`Cập nhật lần cuối: ${text(payload.generated_at)}`]];
  sheet.getRange("A2:I2").format = {
    fill: "#E2E8F0",
    font: { color: "#334155", italic: true },
    horizontalAlignment: "center",
  };

  sheet.getRange("A3:E3").values = [["Tổng lượt", "Đạt", "Không đạt", "Cần review", "Đã đếm"]];
  const lastRow = 5 + records.length;
  const dataStart = 6;
  const dataEnd = Math.max(dataStart, lastRow);
  sheet.getRange("A4:E4").formulas = [[
    records.length ? `=COUNTA(A${dataStart}:A${dataEnd})` : "=0",
    records.length ? `=COUNTIF(C${dataStart}:C${dataEnd},\"PASS\")` : "=0",
    records.length ? `=COUNTIF(C${dataStart}:C${dataEnd},\"FAIL\")` : "=0",
    records.length ? `=COUNTIF(C${dataStart}:C${dataEnd},\"UNKNOWN\")` : "=0",
    records.length ? `=SUM(F${dataStart}:F${dataEnd})` : "=0",
  ]];
  sheet.getRange("A3:E3").format = {
    fill: "#DBEAFE",
    font: { bold: true, color: "#1E3A8A" },
    horizontalAlignment: "center",
  };
  sheet.getRange("A4:E4").format = {
    fill: "#EFF6FF",
    font: { bold: true, color: "#0F172A", size: 13 },
    horizontalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#BFDBFE" },
  };

  const headers = [[
    "Thời gian", "Sản phẩm", "Kết quả", "Màu phát hiện", "Độ tin cậy",
    "Tăng đếm", "Mã track", "Lý do", "Ghi chú",
  ]];
  sheet.getRange("A5:I5").values = headers;
  sheet.getRange("A5:I5").format = {
    fill: "#1E293B",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };

  if (records.length) {
    const rows = records.map((record) => [
      text(record.time), text(record.product_id), text(record.status), text(record.colors),
      number(record.confidence), number(record.count_increment), text(record.track_id),
      text(record.reason), text(record.detail),
    ]);
    sheet.getRange(`A${dataStart}:I${dataEnd}`).values = rows;
    sheet.getRange(`A${dataStart}:I${dataEnd}`).format = {
      borders: { preset: "all", style: "thin", color: "#CBD5E1" },
      verticalAlignment: "center",
    };
    sheet.getRange(`E${dataStart}:E${dataEnd}`).format.numberFormat = "0.0%";
    sheet.getRange(`F${dataStart}:F${dataEnd}`).format.horizontalAlignment = "center";
    sheet.getRange(`C${dataStart}:C${dataEnd}`).format.horizontalAlignment = "center";
    sheet.getRange(`I${dataStart}:I${dataEnd}`).format.wrapText = true;
    sheet.getRange(`C${dataStart}:C${dataEnd}`).conditionalFormats.add("containsText", {
      text: "PASS", format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
    });
    sheet.getRange(`C${dataStart}:C${dataEnd}`).conditionalFormats.add("containsText", {
      text: "FAIL", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
    });
    sheet.getRange(`C${dataStart}:C${dataEnd}`).conditionalFormats.add("containsText", {
      text: "UNKNOWN", format: { fill: "#FEF3C7", font: { color: "#92400E", bold: true } },
    });
  }

  const widths = { A: 13, B: 16, C: 14, D: 38, E: 14, F: 12, G: 12, H: 18, I: 54 };
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}1:${column}${dataEnd}`).format.columnWidth = width;
  }
  sheet.getRange(`A5:I${dataEnd}`).format.rowHeight = 21;
  sheet.freezePanes.freezeRows(5);

  await fs.mkdir(path.dirname(payload.output_path), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(payload.output_path);
}

try {
  const payload = await readPayload();
  await buildWorkbook(payload);
  process.stdout.write(JSON.stringify({ output_path: payload.output_path }));
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
}
