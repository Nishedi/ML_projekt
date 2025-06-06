import pandas as pd

with open('metrics.csv', 'r') as f:
    lines = f.readlines()

lines_replaced = []
for line in lines:
    line = line.replace(';', '&')
    line = line.replace('_', ' ')
    line = line.strip()
    lines_replaced.append(line)

chart = []
chart.append("\\begin{table}")
chart.append("\\centering")
chart.append("\\resizebox{\\textwidth}{!}{")
chart.append("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}")
chart.append("\\hline")
for line in lines_replaced:
    chart.append(line+"\\\\ \hline")
chart.append("\\end{tabular}")
chart.append("}")
chart.append("\\caption{Porównanie metryk jakości klasyfikatorów.}")
chart.append("\\label{tab:metrics_comp}")
chart.append("\\end{table}")

with open('table-metrics.tex', 'w') as f:
    for line in chart:
        f.write(line + "\n")

with open('metrixTime.csv', 'r') as f:
    lines = f.readlines()

lines_replaced = []
for line in lines:
    line = line.replace(';', '&')
    line = line.replace('_', ' ')
    line = line.strip()
    lines_replaced.append(line)

chart = []
chart.append("\\begin{table}")
chart.append("\\centering")
chart.append("\\resizebox{\\textwidth}{!}{")
chart.append("\\begin{tabular}{|c|c|c|}")
chart.append("\\hline")
for line in lines_replaced:
    chart.append(line+"\\\\ \hline")
chart.append("\\end{tabular}")
chart.append("}")
chart.append("\\caption{Porównanie czasasów uczenia i predykcji klasyfikatorów.}")
chart.append("\\label{tab:metrics_time_comp}")
chart.append("\\end{table}")

with open('table-metrics-time.tex', 'w') as f:
    for line in chart:
        f.write(line + "\n")
