function loadAttendanceData() {
    fetch('attendance.csv')
        .then(response => response.text())
        .then(data => {
            const allRows = data.trim().split('\n');
            const headers = allRows[0].split(',');
            const rows = allRows.slice(1);

            const subjectsSet = new Set();
            const roomsSet = new Set();
            const attendanceData = [];

            rows.forEach(row => {
                const cols = row.split(',');
                const rowData = {};
                headers.forEach((header, index) => {
                    rowData[header.trim()] = cols[index] ? cols[index].trim() : '';
                });
                subjectsSet.add(rowData['Subject']);
                roomsSet.add(rowData['Room']);
                attendanceData.push(rowData);
            });

            populateDropdown('subject-select', Array.from(subjectsSet), 'All Subjects');
            populateDropdown('room-select', Array.from(roomsSet), 'All Rooms');
            displayTable(attendanceData);

            // Event listeners for subject and room selection
            document.getElementById('subject-select').addEventListener('change', function() {
                filterAndDisplayTable(attendanceData);
            });
            document.getElementById('room-select').addEventListener('change', function() {
                filterAndDisplayTable(attendanceData);
            });
        })
        .catch(error => console.error('Error loading CSV file:', error));
}

function populateDropdown(selectId, items, defaultOption) {
    const selectElement = document.getElementById(selectId);
    selectElement.innerHTML = `<option value="">${defaultOption}</option>`;
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });
}

function filterAndDisplayTable(attendanceData) {
    const selectedSubject = document.getElementById('subject-select').value;
    const selectedRoom = document.getElementById('room-select').value;

    let filteredData = attendanceData;

    if (selectedSubject) {
        filteredData = filteredData.filter(row => row['Subject'] === selectedSubject);
    }

    if (selectedRoom) {
        filteredData = filteredData.filter(row => row['Room'] === selectedRoom);
    }

    displayTable(filteredData);
}

function displayTable(data) {
    const tableContainer = document.getElementById('table-container');
    tableContainer.innerHTML = '';

    if (data.length === 0) {
        tableContainer.innerHTML = '<p class="text-center">No attendance records found.</p>';
        return;
    }

    const table = document.createElement('table');
    table.classList.add('table', 'table-striped', 'table-bordered');

    const headers = Object.keys(data[0]);
    const thead = document.createElement('thead');
    const trHead = document.createElement('tr');

    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    data.forEach(rowData => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = rowData[header];
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    tableContainer.appendChild(table);
}

document.addEventListener('DOMContentLoaded', loadAttendanceData);
