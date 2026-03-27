with open(r'D:\Aprogress\Langchain\index.html', 'r', encoding='utf-8') as f:
    content = f.read()
print('Total length:', len(content))
for kw in ['renderTasksPage', 'renderKBPage', 'renderAgentsPage', 'setTheme', 'location.hash', 'nav-tabs', 'task-create']:
    pos = content.find(kw)
    if pos >= 0:
        print(f'Found "{kw}" at char {pos}')
    else:
        print(f'NOT FOUND: {kw}')
for kw in ['const S =', 'rag_tid', 'btn-reset', 'btn-cost']:
    pos = content.find(kw)
    if pos >= 0:
        print(f'Found OLD "{kw}" at char {pos}')
