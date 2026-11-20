from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import hashlib,json,re,ssl

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'flight_control_original_restored_2026-10-25/webpage'
GRAPH_TEXT='''基于品质对象族构建统一飞控控制器 品质模型对象库 个对象 维品质相关参数 响应 阻尼 时延 特权强化学习教师 个专用策略 个通过项目门限 普通观测 特权信息 每对象 万步 多教师蒸馏 统一学生控制器 学生运行 教师标注 累计数据 再训练 目标机型适配 仿真 个适配 个留出 步 适配误差 部署反馈 观测与历史 学生策略 力速率动作 执行机构与对象 0123456789,. /+→–() L OOD Actor Critic PPO BC DAgger tiny small DHC-6 ARX Adam deg/s ：·（）'''
CONTEXT=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')
def font(text):
    query=urlencode({'family':'Noto Sans SC','display':'swap','text':text})
    with urlopen(Request('https://fonts.googleapis.com/css2?'+query,headers={'User-Agent':'Mozilla/5.0'}),context=CONTEXT,timeout=30) as r:
        css=r.read().decode()
    url=re.search(r'url\(([^)]+)\)',css).group(1)
    with urlopen(url,context=CONTEXT,timeout=30) as r:return r.read(),url
def main():
    target=ROOT/'webpage/vendor/fonts';target.mkdir(parents=True,exist_ok=True)
    current=ROOT/'webpage/dhc6_report.html'
    text=BeautifulSoup((current if current.exists() else ORIGINAL/'dhc6_report.html').read_text(),'html.parser').get_text()+GRAPH_TEXT
    chars=sorted({c for c in text if ord(c)>127})
    chunks=[''.join(chars[i:i+400]) for i in range(0,len(chars),400)]
    def get(item):
        i,chunk=item;data,url=font(chunk);sha=hashlib.sha256(data).hexdigest();name=f'noto-sc-{sha[:12]}.ttf'
        (target/name).write_bytes(data)
        rule='@font-face{font-family:PaperChinese;src:url("fonts/'+name+'") format("truetype");font-display:swap;unicode-range:'+','.join(f'U+{ord(c):X}' for c in chunk)+';}'
        return rule,{'file':name,'sha256':sha,'source':url,'characters':len(chunk)}
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(get,enumerate(chunks)))
    (ROOT/'webpage/vendor/paper-fonts.css').write_text('\n'.join(x[0] for x in result)+'\nbody,h1.title,h1,h2,h3,h4,.toc,table.bt,table.bt caption,table.bt th,table.bt td,.abstract b,.keywords{font-family:PaperChinese,Georgia,serif;}\n')
    graph_chars=set(GRAPH_TEXT+'轮'+''.join(chr(i) for i in range(32,127)))
    graph,url=font(''.join(sorted(graph_chars)))
    (ROOT/'paper_source/diagram-font.ttf').write_bytes(graph)
    (ROOT/'revision/font_manifest.json').write_text(json.dumps({'web_subsets':[x[1] for x in result],'graph_font_source':url,'graph_font_sha256':hashlib.sha256(graph).hexdigest(),'license':'SIL OFL 1.1'},indent=2)+'\n')
    print('Prepared',len(result),'web font subsets and one diagram font')
if __name__=='__main__':main()
