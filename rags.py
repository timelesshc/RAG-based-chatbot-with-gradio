from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

from llm import MyLLM

# https://docs.ragas.io/

llm = MyLLM()
llm.load_knowledge()

response = llm.invoke("请假流程", "人事管理流程.docx")

truth = '''1.普通员工申请请假/加班/调休：由直属领导、事业部总监、人事行政主管审批。抄送考勤组、请假部门、人事行政经理；
2.主管及以上人员申请请假/调休/请假≥3天：由直属领导、事业部总监、人事行政经理、总经理审批。抄送考勤组、请假部门、人事行政主管、股东会成员；
3.原则上所有请假/加班/调休必须在钉钉系统进行申请并批准后方可执行；因紧急特殊原因请假/调休，可以电话获得部门经理同意后24小时内申请，逾期未申请相关审批人员可不予准假，否则视为旷工。
4.常规请假需提前1个工作日在钉钉系统进行申请，请假3天及以上人员需提前3天申请，并注明请假时间、类型、事由，未按要求提前申请可不予准假，请假审批通过后方可休假，否则按旷工处理；'''

questions = [response['input']] # 问题
answers = [response['answer']] # 回答
contexts = [[doc.page_content for doc in response['context']]] # 文档
ground_truth = [truth] #真实答案

data_samples = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

dataset = Dataset.from_dict(data_samples)

'''
    与评估Retrieval 检索器相关的指标如下:
        - Context Precision  question和contexts   问题和检索到的文档
        - Context Recall。    ground truth和contexts 真实答案和检索到的文档
    与评估Generation 答案相关的指标如下:
        - Faithfulness。     answer和contexts     回答和检索到的文档
        - Answer Relevancy。 answer和question     回答和问题
'''

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)
print(result)

