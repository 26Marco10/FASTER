from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import PageBreak, Image, KeepTogether
from reportlab.lib.units import inch
import pandas as pd
import os
import io
import matplotlib.pyplot as plt



def generate_report(all_results, output_dir=None, output_filename=None):
    """
    Generate a PDF report with tables and charts for the sentiment analysis results.
    :param all_results: List of dictionaries containing the evaluation results for each model
    :param output_dir: Directory where the output file will be saved
    :param output_filename: Name of the output file
    """
    df = pd.DataFrame(all_results)
    if output_dir is None:
        output_dir = "/app/reports"
    if output_filename is None:
        output_filename = f"{output_dir}/full_report.pdf"
    else:
        output_filename = f"{output_dir}/{output_filename}"

    os.makedirs(output_dir, exist_ok=True)
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---- Prima pagina: titolo centrato ----
    title_style = styles['Title']
    title_style.fontSize = 24
    page_height = A4[1]
    vertical_space = (page_height - 2 * title_style.fontSize) / 2 - 1*inch
    elements.append(Spacer(1, vertical_space))
    elements.append(Paragraph("FASTER REPORT (SENTIMENT ANALYZER)", title_style))
    elements.append(Spacer(1, vertical_space))
    elements.append(PageBreak())

    # ---- Prepara i dati ----
    df['Model_Base'] = df['Model'].str.split('(').str[0].str.strip()
    df['Preprocessing_Method'] = df['Model'].str.extract(r'\((.*?)\)', expand=False)
    df = df.sort_values(['Category', 'Model_Base', 'Preprocessing_Method'])

    def truncate_3_dec(val):
        return f"{val:.3f}"

    # Raggruppa per Categoria e Modello Base
    groups = list(df.groupby(['Category', 'Model_Base']))
    total_groups = len(groups)

    # ---- Per ogni gruppo: tabella + grafici su una pagina ----
    for idx, ((category, model_base), group) in enumerate(groups):
        group_elements = []
        # Intestazioni di sezione
        group_elements.append(Paragraph(f"Categoria: {category}", styles['Heading1']))
        group_elements.append(Paragraph(f"Modello: {model_base}", styles['Heading2']))
        group_elements.append(Spacer(1, 0.15*inch))

        # Costruzione della tabella
        table_data = [
            ['Preprocessing', 'Accuracy', 'Precision', 'Recall', 'F1', 
             'Time (s)', 'Mem (MG)', 'CPU (%)']
        ]
        for _, row in group.iterrows():
            table_data.append([
                row['Preprocessing_Method'],
                truncate_3_dec(row['Accuracy']),
                truncate_3_dec(row['Precision']),
                truncate_3_dec(row['Recall']),
                truncate_3_dec(row['F1_score']),
                f"{row['Execution_time']:.3f}",
                f"{row['Memory_usage_mb_during_test']:.3f}",
                f"{row['Cpu_percent_during_test']:.3f}"
            ])

        table = Table(table_data, repeatRows=1, colWidths=[1.8*inch] + [0.7*inch]*7)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2F5496')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BOX', (0,0), (-1,-1), 0.5, colors.black),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        group_elements.append(table)
        group_elements.append(Spacer(1, 0.3*inch))

        # ---- Creazione dei grafici in griglia 2x2 (senza suptitle) ----
        fig_height = 7  # Altezza della figura regolata per 2 righe di grafici
        fig = plt.figure(figsize=(12, fig_height), dpi=130)
        # Il suptitle è stato rimosso per non mostrare "Performance - ..." su ogni pagina

        palette = ['#2F5496', '#C65911', '#2D9D7F', '#6B3FA0']
        metrics = ['Accuracy', 'Execution_time', 'Memory_usage_mb_during_test', 'Cpu_percent_during_test']
        
        # Creazione della griglia 2x2 con maggiore spazio tra le righe
        gs = fig.add_gridspec(2, 2, hspace=1, wspace=0.3)
        axes = gs.subplots().flatten()
        
        for i, (metric, ax) in enumerate(zip(metrics, axes)):
            sorted_group = group.sort_values(metric, ascending=False)
            bars = ax.bar(
                sorted_group['Preprocessing_Method'],
                sorted_group[metric],
                color=palette[i],
                edgecolor='white',
                width=0.5,
                zorder=3
            )
            ax.set_title(metric, fontsize=12, pad=10)
            # Etichette sull'asse x: font ridotto e ruotate verticalmente
            ax.tick_params(axis='x', rotation=90, labelsize=8)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_ylim(0, sorted_group[metric].max() * 1.15)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: truncate_3_dec(x)))
            ax.grid(axis='y', alpha=0.3, zorder=0)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        plt.subplots_adjust(top=0.92, bottom=0.15, left=0.1, right=0.95)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=140)
        plt.close()
        buf.seek(0)  # Riavvolge il buffer per la lettura

        img_width = 7.5 * inch
        img_height = 5 * inch
        group_elements.append(Image(buf, width=img_width, height=img_height))
        group_elements.append(Spacer(1, 0.4*inch))

        # Mantiene insieme tutti gli elementi della sezione per evitare spezzature
        elements.append(KeepTogether(group_elements))
        if idx < total_groups - 1:
            elements.append(PageBreak())

    doc.build(elements)
    print(f"Report generato: {output_filename}")
