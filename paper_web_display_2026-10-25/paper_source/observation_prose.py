"""Describe the observation contract concisely and put its inventory in a table."""
from bs4 import BeautifulSoup


def rewrite_observation_prose(soup):
    paragraph = next(p for p in soup.find_all('p')
                     if p.get_text().startswith('观测按三段拼接'))
    paragraph['id'] = 'observation-inputs'
    paragraph.clear()
    paragraph.append(BeautifulSoup(
        '学生根据任务指令、角速度反馈和控制历史生成动作。输入信号经归一化后组成 176 维观测向量，见表 99。',
        'html.parser'))
    table = BeautifulSoup('''
<div class="parameter-table-wrap" id="observation-table-wrap">
<table class="bt observation-table" id="observation-table">
<caption>表 99　学生控制器的观测组成</caption>
<thead><tr><th scope="col">输入部分</th><th scope="col">信号内容</th><th scope="col">维数</th></tr></thead>
<tbody>
<tr><td>当前观测</td><td>指令、目标角速度、实际角速度、跟踪误差、误差积分、实际角速度与目标角速度的导数、请求力、施加力。</td><td>9</td></tr>
<tr><td>观测历史</td><td>过去 32 步的指令、目标角速度、实际角速度、跟踪误差和请求力，共 5 个通道。</td><td>160</td></tr>
<tr><td>动作历史</td><td>最近 7 步的归一化控制动作。</td><td>7</td></tr>
</tbody></table></div>''', 'html.parser').find('div')
    paragraph.insert_after(table)
