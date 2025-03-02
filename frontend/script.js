let currentTableId = null;
let currentQuery = null;

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('loadSample').addEventListener('click', loadSample);
    document.getElementById('predict').addEventListener('click', predictQuery);
    document.getElementById('executeQuery').addEventListener('click', executeQuery);
});

async function loadSample() {
    try {
        const response = await fetch('/get_random_sample', { method: 'POST' });
        const data = await response.json();
        
        // Store table ID for later use
        currentTableId = data.table_id;
        
        // Display the question
        document.getElementById('question').textContent = data.question;
        
        // Create and display the table
        const table = document.getElementById('table');
        table.innerHTML = ''; // Clear existing table
        
        // Create header row
        const headerRow = document.createElement('tr');
        data.table.header.forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);
        
        // Create data rows
        data.table.rows.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cellData => {
                const td = document.createElement('td');
                td.textContent = cellData || ''; // Handle null/undefined values
                tr.appendChild(td);
            });
            table.appendChild(tr);
        });
        
        // Clear previous results
        document.getElementById('generatedQuery').textContent = '';
        document.getElementById('queryResult').textContent = '';
        
    } catch (error) {
        console.error('Error:', error);
    }
}

async function predictQuery() {
    console.log("predictQuery function called"); // Log to check if the function is called
    try {
        const question = document.getElementById('customQuestionInput').value || 
                        document.getElementById('question').textContent;
        
        if (!question) {
            alert('Please load a sample or enter a question');
            return;
        }
        
        const tableData = {
            question: question,
            table: {
                header: Array.from(document.querySelectorAll('#table th')).map(th => th.textContent),
                rows: Array.from(document.querySelectorAll('#table tr')).slice(1)
                    .map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent))
            }
        };
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(tableData)
        });
        
        const data = await response.json();
        console.log("Response Data:", data); // Log the response data
        currentQuery = data.query;
        document.getElementById('modelOutput').textContent = JSON.stringify(data.model_output, null, 2);
        document.getElementById('generatedQuery').textContent = JSON.stringify(data.query, null, 2);
        
    } catch (error) {
        console.error('Error:', error);
    }
}

async function executeQuery() {
    try {
        if (!currentQuery || !currentTableId) {
            alert('Please generate a query first');
            return;
        }
        
        const response = await fetch('/execute_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                table_id: currentTableId,
                query: currentQuery
            })
        });
        
        const data = await response.json();
        document.getElementById('queryResult').textContent = JSON.stringify(data.result, null, 2);
        
    } catch (error) {
        console.error('Error:', error);
    }
}