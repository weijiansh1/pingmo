from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag
import re,json,os,subprocess,hashlib,shutil
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
WEB=BASE/'paper_web_display_2026-10-25/webpage'
ORIGINAL=BASE/'flight_control_original_restored_2026-10-25/webpage'
SOURCE=WEB/'dhc6_report.html'
soup=BeautifulSoup(SOURCE.read_text(),'html.parser')
manifest={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'figures':[],'tables':[],'equations':[]}

def esc(s):
    special={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    symbols={'θ':r'\theta','τ':r'\tau','ω':r'\omega','ζ':r'\zeta','λ':r'\lambda','γ':r'\gamma','δ':r'\delta','μ':r'\mu','π':r'\pi','ε':r'\epsilon','ρ':r'\rho','η':r'\eta','φ':r'\phi','±':r'\pm','×':r'\times','→':r'\rightarrow','←':r'\leftarrow','≥':r'\ge','≤':r'\le','−':'-','²':'{}^2','³':'{}^3','ṗ':r'\dot p','Â':r'\widehat A','⁻':'{}^-','¹':'{}^1','∈':r'\in'}
    return ''.join(special.get(c,('\\ensuremath{'+symbols[c]+'}') if c in symbols else c) for c in s if c not in '\u200b\ufeff')

def inline(n):
    if isinstance(n,NavigableString):return esc(str(n))
    if not isinstance(n,Tag):return ''
    classes=n.get('class',[])
    if 'math-inline' in classes or 'katex' in classes:
        a=n.find('annotation',encoding='application/x-tex')
        if a:return r'\('+a.get_text()+r'\)'
    if n.name in ['script','style']:return ''
    if n.name=='br':return r'\newline{} '
    txt=''.join(inline(x) for x in n.children)
    if n.name in ['b','strong']:return r'\textbf{'+txt+'}'
    if n.name in ['i','em']:return r'\textit{'+txt+'}'
    if n.name=='sub':return r'\textsubscript{'+txt+'}'
    if n.name=='sup':return r'\textsuperscript{'+txt+'}'
    if n.name=='code':return r'\texttt{'+txt+'}'
    if n.name=='a' and n.get('href','').startswith('#'):
        return r'\hyperlink{'+n['href'][1:]+'}{'+txt+'}'
    if n.name=='a' and n.get('href','').startswith('https://'):
        return r'\href{'+n['href'].replace('%',r'\%')+'}{'+txt+'}'
    return txt

def target(n):return (r'\hypertarget{'+n['id']+'}{}\label{'+n['id']+'}\n') if n.get('id') else ''

figure_specs=[]
for i,f in enumerate(soup.select('main figure'),1):
    rel=f.img['src'].split('?')[0];src=WEB/rel
    if not src.exists():src=ORIGINAL/rel
    raw=src.read_text();sv=BeautifulSoup(raw,'xml').find('svg');vb=[float(x) for x in sv['viewBox'].split()]
    w,h=vb[2:];dest=ROOT/f'figures/fig{i:02d}.pdf'
    figure_specs.append({'url':'http://127.0.0.1:8320/webpage/'+f.img['src'],'width':w,'height':h,'path':str(dest)})
    manifest['figures'].append({'number':i,'svg':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'pdf':str(dest),'caption':f.figcaption.get_text(' ',strip=True)})

def make_figures():
    env=dict(os.environ);env['LD_LIBRARY_PATH']='/tmp/pw-libs/usr/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
    cli=['npx','--yes','--package','node@22','--package','@playwright/cli','playwright-cli','-s=latex-figure-conversion']
    config=BASE/'paper_web_proxy_2026-10-25/output/playwright/original_display.config.json'
    def run(args,timeout=55):
        p=subprocess.run(cli+args,env=env,capture_output=True,text=True,timeout=timeout)
        if p.returncode or '### Error' in p.stdout:raise RuntimeError(p.stdout[-1600:]+p.stderr[-600:])
        return p.stdout
    run(['open','about:blank','--config',str(config)])
    code='''async(page)=>{
await page.goto('http://127.0.0.1:8320/webpage/dhc6_report.html',{waitUntil:'domcontentloaded',timeout:15000});
await page.addStyleTag({content:'html,body{margin:0!important;padding:0!important;background:white!important;}body>img{display:block;margin:0!important;max-width:none!important;}'});
for(const s of SPECS){
 await page.setViewportSize({width:Math.ceil(s.width*4/3),height:Math.ceil(s.height*4/3)});
 await page.evaluate(async s=>{document.body.innerHTML='';const im=new Image();im.style.width=s.width+'pt';im.style.height=s.height+'pt';document.body.appendChild(im);im.src=s.url;await im.decode();},s);
 await page.pdf({path:s.path,width:(s.width/72)+'in',height:(s.height/72)+'in',margin:{top:0,right:0,bottom:0,left:0},printBackground:true,displayHeaderFooter:false,preferCSSPageSize:false});
}
return {converted:SPECS.length};}'''.replace('SPECS',json.dumps(figure_specs))
    try:print(run(['run-code',code]).split('### Ran Playwright code')[0],flush=True)
    finally:run(['close'])

PREAMBLE=r'''\documentclass[UTF8,a4paper,11pt,fontset=fandol,scheme=plain]{ctexart}
\usepackage[margin=23mm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb,bm,graphicx,booktabs,tabularx,array,caption,float,placeins}
\usepackage{xcolor,enumitem,fancyhdr}
\usepackage[unicode,colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue!45!black]{hyperref}
\setmainfont{texgyretermes-regular.otf}[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\linespread{1.18}
\setlength{\emergencystretch}{2em}
\setlength{\textfloatsep}{10pt}
\raggedbottom
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.8}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
\setlength{\intextsep}{12pt}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\refname}{参考文献}
\captionsetup{font=small,labelsep=quad,justification=justified,singlelinecheck=false}
\captionsetup[table]{position=top,skip=6pt}
\ctexset{section={format=\Large\heiti,beforeskip=18pt,afterskip=8pt},subsection={format=\large\heiti,beforeskip=13pt,afterskip=6pt},subsubsection={format=\normalsize\heiti,beforeskip=10pt,afterskip=5pt}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 基于飞行品质的飞控大模型构建与滚转通道验证}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\newsavebox{\EqBox}
\newcommand{\fitmath}[1]{\sbox{\EqBox}{$\displaystyle #1$}\ifdim\wd\EqBox>0.92\linewidth\resizebox{0.92\linewidth}{!}{\usebox{\EqBox}}\else\usebox{\EqBox}\fi}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\begin{document}
'''

def render(n):
    if not isinstance(n,Tag):return ''
    if n.name=='h1':return '{\centering\LARGE\heiti '+inline(n)+r'\par}'+'\n\vspace{6pt}\n'
    if n.name in ['h2','h3','h4']:
        text=n.get_text(' ',strip=True)
        if n.get('id')=='refs':return '\\FloatBarrier\n'
        if n.get('id')=='appendix-history':
            prefix='\\FloatBarrier\n\\appendix\n\\ctexset{section/name={附录~,},section/number=\\Alph{section}}\n'
        else:prefix='\\FloatBarrier\n' if n.name=='h2' else ''
        text=re.sub(r'^(?:附录\s*[A-Z]|[A-Z](?:\.\d+)+|\d+(?:\.\d+)*)\s*','',text)
        number=re.match(r'^(?:附录\s*)?([0-9A-Z]+(?:\.\d+)*)',n.get_text(' ',strip=True))
        depth=len(number.group(1).split('.')) if number else 1
        command={1:'section',2:'subsection',3:'subsubsection'}.get(depth,'subsubsection')
        return prefix+'\\'+command+'{'+esc(text)+'}\n'+target(n)
    if n.name=='p':return target(n)+inline(n)+'\n\n'
    if n.name=='figure':
        idx=list(soup.select('main figure')).index(n)+1
        cap=re.sub(r'^图\s*\d+\s*','',inline(n.figcaption))
        scale=next((float(c[1:])/100 for c in n.get('class',[]) if re.fullmatch(r'w\d+',c)),1.0)
        if idx==7:scale=.60
        figure_width=f'{scale:.2f}'+r'\linewidth'
        return (r'\begin{figure}[!htbp]\centering'+'\n'+target(n)+r'\includegraphics[width='+figure_width+r',height=.65\textheight,keepaspectratio]{figures/fig'+f'{idx:02d}'+'.pdf}\n'+r'\caption{'+cap+'}\n'+r'\end{figure}'+'\n')
    if n.name=='table':
        rows=n.select('tr');nc=len(rows[0].find_all(['th','td'],recursive=False));idx=len(manifest['tables'])+1
        cap=re.sub(r'^表\s*\d+\s*','',inline(n.caption))
        weights=[1]*nc
        if n.get('id')=='model-parameter-table':weights=[.6,1.05,1.8,.55]
        elif n.get('id')=='observation-table':weights=[.65,1.95,.4]
        elif n.get('id')=='sampling-protocol-table':weights=[.42,1.62,.96]
        elif nc==2:weights=[.35,1.65]
        elif nc==8:weights=[1.5,1.05,1.2,1.85,.6,.6,.6,.6]
        spec=''.join(r'>{\hsize='+str(w)+r'\hsize\linewidth=\hsize\raggedright\arraybackslash}X' for w in weights)
        out=r'\begin{table}[!htbp]\centering'+'\n'+target(n)+r'\caption{'+cap+'}\n'+r'\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.18}'+'\n'+r'\begin{tabularx}{\linewidth}{'+spec+'}\toprule\n'
        for i,row in enumerate(rows):
            cells=row.find_all(['th','td'],recursive=False)
            out+=' & '.join(inline(c).strip() for c in cells)+r' \\'+'\n'
            if i==0:out+=r'\midrule'+'\n'
        out+=r'\bottomrule\end{tabularx}\end{table}'+'\n'
        manifest['tables'].append({'number':idx,'caption':n.caption.get_text(' ',strip=True),'cells':[[c.get_text(' ',strip=True) for c in r.find_all(['th','td'],recursive=False)] for r in rows]})
        return out
    if n.name=='div' and 'eq' in n.get('class',[]):
        a=n.find('annotation',encoding='application/x-tex');tex=a.get_text();manifest['equations'].append(tex)
        return target(n)+r'\begin{equation}\fitmath{'+tex+r'}\end{equation}'+'\n'
    if n.name=='ol' and 'refs' in n.get('class',[]):
        return r'\begin{thebibliography}{99}'+'\n'+''.join(r'\bibitem{ref'+str(i)+'} '+inline(li)+'\n' for i,li in enumerate(n.find_all('li',recursive=False),1))+r'\end{thebibliography}'+'\n'
    if n.name in ['ul','ol']:
        env='itemize' if n.name=='ul' else 'enumerate'
        return '\\begin{'+env+'}[nosep]\n'+''.join('\\item '+inline(li)+'\n' for li in n.find_all('li',recursive=False))+'\\end{'+env+'}\n'
    if n.name=='div' and 'abstract' in n.get('class',[]):
        text=re.sub(r'^\\textbf\{摘要\}\s*|^摘要\s*','',inline(n))
        return r'\begin{abstract}\noindent '+text+r'\end{abstract}'+'\n'
    if n.name=='div' and 'meta' in n.get('class',[]):return '{\centering\small '+inline(n)+r'\par}'+'\n\vspace{8pt}\n'
    if n.name=='div' and 'keywords' in n.get('class',[]):return '\\noindent '+inline(n)+'\n\vspace{10pt}\n'
    if n.name in ['div','section','span']:return target(n)+''.join(render(c) for c in n.children if isinstance(c,Tag))
    return ''

body=''.join(render(n) for n in soup.main.children if isinstance(n,Tag))
body=re.sub(r'\\n(?![A-Za-z])','\n',body)
body=body.replace(chr(11),chr(92)+'v').replace(chr(9)+'oprule',chr(92)+'toprule')
(ROOT/'report.tex').write_text(PREAMBLE+body+'\n\\FloatBarrier\n\\end{document}\n')
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert len(manifest['figures'])==22 and len(manifest['tables'])==16 and len(manifest['equations'])==13
if not all(Path(s['path']).is_file() for s in figure_specs):make_figures()
for iteration in range(2):
    p=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=ROOT,capture_output=True,text=True)
    (ROOT/f'compile-{iteration+1}.txt').write_text(p.stdout+p.stderr)
    if p.returncode:print(p.stdout[-5000:],flush=True);raise SystemExit(p.returncode)
    print(p.stdout[-400:],flush=True)
dest=BASE/'基于飞行品质的飞控大模型构建与滚转通道验证_LaTeX版.pdf'
shutil.copy2(ROOT/'report.pdf',dest)
print('PDF',dest,flush=True)
