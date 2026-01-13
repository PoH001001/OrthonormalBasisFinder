const column_inputs = document.querySelectorAll(".column");
const row_inputs = document.querySelectorAll(".row");
const span_texts = document.getElementsByClassName("number");
const sup_texts = document.getElementsByClassName("exponent");

column_inputs.forEach(input => {
  input.addEventListener("input", (event) => {
    const value = event.target.value;

    for (let span of span_texts) {
      span.textContent = value;
    }

    for (let sup of sup_texts) {
      sup.textContent = value;
    }
  });
});

//Gets the current row/column values
function createTable() {
    const row = document.querySelector(".row");
    const column = document.querySelector(".column");
    const grid = document.querySelector(".Matrix");

    grid.innerHTML = "";

    const table_Input = document.createElement("table");
    const labels = document.createElement("thead");
    const labelRow = document.createElement("tr");

    for(let j = 0; j < column.value; j++) {
      const newHeader = document.createElement("th");

      newHeader.textContent = `v`+`<sup>${j+1}</sup>`;
      newHeader.innerHTML = newHeader.textContent;
      labelRow.appendChild(newHeader);
    }
    labels.appendChild(labelRow);
    table_Input.appendChild(labels);

    for (let i = 0; i < row.value; i++) {
        const newRow = document.createElement("tr");
        
        for (let j = 0; j < column.value; j++) {
            const newColumn = document.createElement("td");

            const input = document.createElement("input");
            input.type = "number";
            input.className = "matrixInput";

            newColumn.appendChild(input);
            newRow.appendChild(newColumn);
            console.log("Input Created");
        }
        table_Input.appendChild(newRow);
    }
    grid.appendChild(table_Input);
}

//start at a 2xx2
const rowInput = document.querySelector(".row");
const colInput = document.querySelector(".column");
rowInput.value = 2;
colInput.value = 2;
createTable();

column_inputs.forEach(input => {
  input.addEventListener("input", createTable);
});
row_inputs.forEach(input => {
  input.addEventListener("input", createTable);
});

let history_data = [];
let index = 0;

const history_box = document.getElementById("history");
const no_calculations = document.getElementsByClassName("NoCalculations");

function addToHistory(data) {
  if (index === 0 && no_calculations.length > 0) {
    no_calculations[0].remove();
  }

  let entry = data;
  history_data.push(entry);

  let li = document.createElement("li");
  li.innerHTML = entry;
  history_box.appendChild(li);

  index++;
} 

//Actual Computations
function SendRequest() {
  const matrix = document.querySelector(".Matrix table");
    const rows = matrix.rows.length;
    const cols = matrix.rows[0].cells.length;

    let matrix_input = [];

    for (let i = 1; i < rows; i++) {
    let row = [];
      for (let j = 0; j < cols; j++) {
        const input = matrix.rows[i].cells[j].querySelector("input");
        row.push(input ? Number(input.value) : 0);
      }
    matrix_input.push(row);
  }

fetch("/Group2", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ "matrix": matrix_input })
})

  .then(response => response.json())
  .then(data => {
    displayResult(data);
  })
  .catch(error => console.error("Error starting stream:", error))
}
const button = document.querySelector(".compute-btn");
button.addEventListener("click", SendRequest);



function displayResult(data) {
  const resultContainer = document.getElementById("output");
  resultContainer.innerHTML = "";

  if (data.result === 0) {
    const errorMsg = document.createElement("div");
    errorMsg.innerText = "The vectors are Linearly Dependent. No Orthonormal Basis exists.";
    resultContainer.appendChild(errorMsg);
    return;
  }

  const header = document.createElement("p");
  header.innerText = "The orthonormal basis vectors are:";
  resultContainer.appendChild(header);

  data.result.forEach((row, index) => {
    const rowDiv = document.createElement("div");

    const formattedRow = row.map(val => {
        val = String(val);
        val = val.replace(/sqrt\(([^)]+)\)\/(\d+)/g, (match, numerator, denominator) => {
            return `<mfrac><msqrt><mn>${numerator}</mn></msqrt><mn>${denominator}</mn></mfrac>`;
        })
        .replace(/sqrt\(([^)]+)\)/g, (match, num) => `<msqrt><mn>${num}</mn></msqrt>`)
        .replace(/^(\d+)$/g, '<mn>$1</mn>');
        
        return `<mspace width="0.3em"/>${val}<mspace width="0.3em"/>`; 
    });


    
    rowDiv.innerHTML = `v${index + 1} = [ <math xmlns="http://www.w3.org/1998/Math/MathML">${formattedRow.join(",")}</math> ]<br>`;
    resultContainer.appendChild(rowDiv);

    addToHistory(rowDiv.innerHTML);


  });
  
  let line = document.createElement("hr");
  history_box.appendChild(line);
}