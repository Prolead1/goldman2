"""Render part5/report.md to a professional PDF using ReportLab Platypus."""
from __future__ import annotations
from html import escape
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'part5'
OUT=BASE/'outputs'
NAVY=colors.HexColor('#172C43')


def fmt(s):
    s=escape(s.strip()).replace('`','')
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\*(.*?)\*',r'<i>\1</i>',s)
    s=s.replace('—','-').replace('−','-').replace('×','x').replace('²','2').replace('ᵢ','i').replace('ₚ','p').replace('Σ','sum ').replace('α','alpha').replace('β','beta').replace('ε','epsilon').replace('√','sqrt').replace('¼','1/4').replace('’',"'").replace('“','"').replace('”','"').replace('§','section ')
    return s


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdf=OUT/'Part_5_Portfolio_Analytics_Report.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=17*mm,title='FE5108 Part 5 - Portfolio Analytics')
    styles={
        'title':ParagraphStyle('T',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=NAVY,spaceAfter=10),
        'section':ParagraphStyle('S',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=NAVY,spaceBefore=12,spaceAfter=5,keepWithNext=True),
        'body':ParagraphStyle('B',fontName='Helvetica',fontSize=8.7,leading=12.5,spaceAfter=7,alignment=TA_LEFT),
        'cell':ParagraphStyle('C',fontName='Helvetica',fontSize=7.5,leading=10),
        'summary':ParagraphStyle('Summary',fontName='Helvetica',fontSize=9.2,leading=13.5,spaceAfter=0),
        'head':ParagraphStyle('H',fontName='Helvetica-Bold',fontSize=7.5,leading=10,textColor=colors.white),
    }
    story=[]; lines=(BASE/'report.md').read_text().splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line.startswith('# '):story.append(Paragraph(fmt(line[2:]),styles['title']));continue
        if line.startswith('## '):story.append(Paragraph(fmt(line[3:]),styles['section']));continue
        if line.startswith('!['):
            name=re.search(r'\(([^)]+)\)',line)
            if name:
                path=OUT/name.group(1)
                if path.exists():
                    from PIL import Image as PILImage
                    with PILImage.open(path) as im: w,h=im.size
                    width=145*mm; height=width*h/w
                    story.append(Image(str(path),width=width,height=height));story.append(Spacer(1,5))
            continue
        if line.startswith('|'):
            raw=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):raw.append(lines[i].strip());i+=1
            raw=[r for r in raw if not re.fullmatch(r'[\s|:\-]+',r)]
            data=[[c.strip() for c in r.strip('|').split('|')] for r in raw]
            if not data:continue
            n=max(len(row) for row in data)
            data=[row+['']*(n-len(row)) for row in data]
            available=174*mm
            widths=([available*.36]+[available*.64/(n-1)]*(n-1)) if n>1 else [available]
            if n==4:widths=[available*.32,available*.21,available*.19,available*.28]
            if n==3:widths=[available*.45,available*.275,available*.275]
            cells=[[Paragraph(fmt(v),styles['head'] if j==0 else styles['cell']) for v in row] for j,row in enumerate(data)]
            tbl=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#B9C6D0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')])]))
            story.append(tbl);story.append(Spacer(1,7));continue
        if line.startswith('**Summary.**'):
            story.append(Table([[Paragraph(fmt(line),styles['summary'])]],colWidths=[174*mm],style=TableStyle([('BOX',(0,0),(-1,-1),.8,NAVY),('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F3F6F9')),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)])))
            story.append(Spacer(1,10));continue
        para=line
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
            para+=' '+lines[i].strip();i+=1
        story.append(Paragraph(fmt(para),styles['body']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#6D7781'))
        canvas.drawString(18*mm,11*mm,'FE5108 | Part 5 | Gross monthly USD returns, 1995-2023')
        canvas.drawRightString(192*mm,11*mm,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(pdf)

if __name__=='__main__':main()
