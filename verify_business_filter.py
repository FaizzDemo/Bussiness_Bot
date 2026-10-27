import sys
sys.path.insert(0, r'c:\Users\Administrator\Desktop\busiiness bot\business-knowledge-ai')
from app import is_business_related_question

print(is_business_related_question('what is bussiness'))
print(is_business_related_question('what is business'))
print(is_business_related_question('tell me a joke'))
print(is_business_related_question('how to do marketing'))
