/* data.js
 * Loads and prepares Titanic datasets directly in the browser.
 * Place train.csv, test.csv, gender_submission.csv alongside this file.
 * Exports: window.TitanicData = { loadData, getCategories }
 */

(function () {
  const CSV_URLS = {
    train: "train.csv",
    test: "test.csv",
    gender: "gender_submission.csv",
  };

  // Basic CSV parser (handles quoted commas)
  function parseCSV(text) {
    const rows = [];
    let row = [];
    let cur = "";
    let inQuotes = false;

    function pushCell() {
      row.push(cur.replace(/^"|"$/g, "").replace(/""/g, '"'));
      cur = "";
    }
    function pushRow() {
      rows.push(row);
      row = [];
    }

    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (c === '"') {
        if (inQuotes && text[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c === "," && !inQuotes) {
        pushCell();
      } else if ((c === "\n" || c === "\r") && !inQuotes) {
        if (cur.length || row.length) pushCell();
        if (row.length) pushRow();
      } else {
        cur += c;
      }
    }
    if (cur.length || row.length) {
      pushCell();
      if (row.length) pushRow();
    }
    const header = rows.shift();
    const objects = rows.map((r) => {
      const o = {};
      header.forEach((h, idx) => (o[h] = r[idx] ?? ""));
      return o;
    });
    return objects;
  }

  async function fetchCSV(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
    const text = await res.text();
    return parseCSV(text);
  }

  function toNum(v) {
    if (v === undefined || v === null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  // Title extraction from Name
  function extractTitle(name) {
    if (!name) return "Unknown";
    const m = name.match(/,\s*([^\.]+)\./);
    if (!m) return "Unknown";
    const raw = m[1].trim();
    const map = {
      "Mlle": "Miss",
      "Ms": "Miss",
      "Mme": "Mrs",
      "Lady": "Royalty",
      "Countess": "Royalty",
      "Dona": "Royalty",
      "Sir": "Royalty",
      "Jonkheer": "Royalty",
      "Don": "Royalty",
      "Rev": "Clergy",
      "Dr": "Dr",
      "Col": "Military",
      "Major": "Military",
      "Capt": "Military",
    };
    return map[raw] || raw;
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
      return sorted[base];
    }
  }

  function median(values) {
    const arr = values.filter((v) => v !== null && v !== undefined).slice().sort((a, b) => a - b);
    if (arr.length === 0) return null;
    return quantile(arr, 0.5);
  }

  function createFareQuartile(fare, q1, q2, q3) {
    if (fare === null) return "Unknown";
    if (fare <= q1) return "Q1 (Lowest)";
    if (fare <= q2) return "Q2";
    if (fare <= q3) return "Q3";
    return "Q4 (Highest)";
  }

  function ageGroup(age) {
    if (age === null) return "Unknown";
    if (age <= 12) return "Child";
    if (age >= 60) return "Senior";
    return "Adult";
  }

  function familySize(sibsp, parch) {
    // Common definition includes passenger themselves; requirement says "from SibSp + Parch".
    // We add +1 to reflect household size including the passenger (common in Titanic analyses).
    const s = (sibsp ?? 0) + (parch ?? 0) + 1;
    if (s <= 1) return "Solo (1)";
    if (s <= 3) return "Small (2–3)";
    if (s <= 5) return "Medium (4–5)";
    return "Large (6+)";
  }

  function coalesceEmbarked(v) {
    return v && v.trim() ? v.trim() : "Unknown";
  }

  function toIntOrNull(x) {
    const n = parseInt(x, 10);
    return Number.isFinite(n) ? n : null;
    }

  async function loadData() {
    // 1) Load CSVs
    const [train, test, gender] = await Promise.all([
      fetchCSV(CSV_URLS.train),
      fetchCSV(CSV_URLS.test),
      fetchCSV(CSV_URLS.gender),
    ]);

    // 2) Merge test + gender_submission Survived labels by PassengerId
    const testSurvivedMap = new Map(gender.map((g) => [toIntOrNull(g.PassengerId), toIntOrNull(g.Survived)]));
    const fullTest = test.map((r) => ({
      ...r,
      Survived: toIntOrNull(testSurvivedMap.get(toIntOrNull(r.PassengerId))),
    }));

    // 3) Combine
    const combinedRaw = [...train, ...fullTest];

    // 4) Convert fields & collect arrays for stats
    const rows = combinedRaw.map((r) => {
      const Age = toNum(r.Age);
      const Fare = toNum(r.Fare);
      return {
        PassengerId: toIntOrNull(r.PassengerId),
        Survived: toIntOrNull(r.Survived),  // 0/1
        Pclass: toIntOrNull(r.Pclass),
        Name: r.Name,
        Sex: r.Sex ? r.Sex.trim().toLowerCase() : "unknown",
        Age,
        SibSp: toIntOrNull(r.SibSp),
        Parch: toIntOrNull(r.Parch),
        Ticket: r.Ticket,
        Fare,
        Cabin: r.Cabin,
        Embarked: coalesceEmbarked(r.Embarked),
        Source: r.hasOwnProperty("Survived") && r.Survived !== "" ? "train_or_test_with_label" : "unknown",
      };
    });

    // 5) Median imputation for Age (across combined data)
    const ageMedian = median(rows.map((r) => r.Age));
    rows.forEach((r) => {
      if (r.Age === null) r.Age = ageMedian;
    });

    // 6) Fare quartiles (compute from available fares)
    const fareVals = rows.map((r) => r.Fare).filter((v) => v !== null).sort((a, b) => a - b);
    const q1 = quantile(fareVals, 0.25);
    const q2 = quantile(fareVals, 0.50);
    const q3 = quantile(fareVals, 0.75);

    // 7) Feature engineering
    rows.forEach((r) => {
      r.Title = extractTitle(r.Name);
      r.AgeGroup = ageGroup(r.Age);
      r.FamilySize = familySize(r.SibSp ?? 0, r.Parch ?? 0);
      r.FareQuartile = createFareQuartile(r.Fare, q1, q2, q3);
    });

    // Some titles are rare — group long tail for clearer plots
    const commonTitles = new Set(["Mr", "Mrs", "Miss", "Master", "Dr", "Military", "Clergy", "Royalty"]);
    rows.forEach((r) => {
      if (!commonTitles.has(r.Title)) r.Title = "Other";
    });

    const fields = [
      "PassengerId","Survived","Pclass","Sex","Age","AgeGroup","SibSp","Parch","FamilySize","Fare","FareQuartile","Embarked","Title"
    ];

    const categories = {
      Pclass: ["1","2","3"],
      Sex: ["male","female"],
      AgeGroup: ["Child","Adult","Senior"],
      FareQuartile: ["Q1 (Lowest)","Q2","Q3","Q4 (Highest)","Unknown"],
      Embarked: ["C","Q","S","Unknown"],
      FamilySize: ["Solo (1)", "Small (2–3)", "Medium (4–5)", "Large (6+)"],
      Title: ["Mr","Mrs","Miss","Master","Dr","Military","Clergy","Royalty","Other"],
    };

    const meta = {
      ageMedian,
      fareQuantiles: { q1, q2, q3 },
      n: rows.length,
      nLabeled: rows.filter(r => r.Survived !== null).length
    };

    return { rows, fields, categories, meta };
  }

  function getCategories() {
    return ["Pclass","Sex","AgeGroup","FareQuartile","Embarked","FamilySize","Title"];
  }

  window.TitanicData = { loadData, getCategories };
})();
