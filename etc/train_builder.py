def split2phase(line):
    stack = list()
    result = list()
    i = 0
    while i < len(line):
        if line[i] == '(':
            stack.append('')
            i += 3
            continue

        if line[i] == ')':
            result.append(stack.pop())
            i += 1
            continue

        for j in range(len(stack)):
            stack[j] += line[i]
        i += 1

    return result

if __name__ == '__main__':
    import os
    path = os.getcwd()
    tree_file_path = os.path.join(path, 'datasets/SST/trees/train.txt')
    out_file_path = os.path.join(path, 'datasets/Diy/sst/intermediates/phase_train.txt')

    with open(tree_file_path, 'r', encoding='utf-8') as f_in, open(out_file_path, 'w', encoding='utf-8') as f_out:
        count = 0
        f_out.write('sentence_id|sentence\n')
        i = 0
        while True:
            input_ = f_in.readline()
            if input_ == '':
                break
            phases = split2phase(input_)

            for phase in phases:
                f_out.write(str(i) + '|')
                i += 1
                out_phase = phase.replace('-LRB-', '(')
                out_phase = out_phase.replace('-RRB-', ')')
                f_out.write(out_phase + '\n')
            count += 1
            print('Current num of sentence:', count)
