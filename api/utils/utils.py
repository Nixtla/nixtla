from pydantic import BaseModel 


def parse_args(args: BaseModel):
    dict_args = args.dict(exclude_none=True)

    outputs_args = []

    for key, value in dict_args.items():
        key = key.replace('_', '-')
        key = f'--{key}'

        outputs_args.append(key)
        outputs_args.append(str(value))

    return outputs_args

if __name__ == '__main__':

    args = BaseModel.construct(arg_1='arg1', arg_2='arg2', 
                               argn='argn', arg_3=None)

    expected = ['--arg-1', 'arg1', '--arg-2', 'arg2', 
                '--argn', 'argn']

    parsed_args = parse_args(args)

    print(f'expected args: {expected}')
    print(f'parsed args: {parsed_args}')

    assert expected == parsed_args
