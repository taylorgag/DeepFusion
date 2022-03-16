

from functools import reduce
import os
import subprocess
import json
from pydot import io


CONSTRUCTOR_FLAG = "constructor"

FALLBACK_FLAG = "fallback"
UINT256_FLAG = "uint256"

ADD_EQU_FLAG = "+="

EQU_FLAG = "="

ADD_FLAG = "+"

SUB_EQU_FLAG = "-="

SUB_FLAG = "-"

SAFEMATH_FLAG = "SAFEMATH"

LIBRARY_FLAG = "library"

ADD_STR_FLAG = "add"

SUB_STR_FLAG = "sub"

DATASET_PATH = "./dataset/"
EXTRACTED_CONTRACT_SUFFIX = "_reentrancy.sol"


EDGE_FLAG = " -> "
LABEL_FLAG = "[label="
CLUSTER_FLAG = "cluster_"


FLAG_FUNC = -2


class extractReen:
    def __init__(self,_contractPath,_json,_filename):
        self.filename = _filename
        self.json = _json
        self.contractPath = _contractPath
        self.callGraphCache = './cache'
        self.result = 0
        self.contractName = self.getContractName()
        try:
            os.mkdir(DATASET_PATH)
        except:
            pass

    def getAllFuncCallGraph(self):

        try:
            subprocess.run("slither " + self.contractPath + " --print call-graph", check=True, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            # print("Failed to generate functions call-graph.")
            pass

    def contractNameToNum(self,_callGraph):

        result = list()
        fileName = [filename for filename in os.listdir(self.callGraphCache) if
                    filename.endswith('.dot') and '.all_contracts.call-graph' in filename]
        # print(fileName)
        for filePath in fileName:
            dotFile = os.path.join(self.callGraphCache, filePath)
            f = io.open(dotFile)
            contractNameDict = dict()
            for line in f.readlines():
                if line.find(CLUSTER_FLAG) != -1:
                    try:
                        temp = line.split(" ")[1]
                        num, contractName = self.splitTemp(temp)
                        contractNameDict[contractName] = num
                    except:
                        continue
                else:
                    continue
            for _list in _callGraph:
                aList = list()
                for func in _list:
                    try:
                        num, funcName = self.splitTempName(func)
                        for item in contractNameDict.items():
                            if item[1] == num:
                                temp = item[0] + "." + funcName
                                aList.append(temp)
                            else:
                                continue
                    except:
                        continue
                result.append(aList)
        return result

    def getCallGraphDot(self):
        fileName = [filename for filename in os.listdir(self.callGraphCache) if
                    filename.endswith('.dot') and '.all_contracts.call-graph' in filename]
        self.funcCallGraph = list()
        edgeList = list()
        for filePath in fileName:
            dotFile = os.path.join(self.callGraphCache, filePath)
            try:
                f = io.open(dotFile)
                for line in f.readlines():
                    if line.find(EDGE_FLAG) != -1:
                        if line.startswith('}') or line.endswith('}'):
                            edgeInfo = list()
                            edgeInfo.append(line.split(EDGE_FLAG)[0][1:])
                            edgeInfo.append(line.split(EDGE_FLAG)[1][:-1])
                            edgeList.append(edgeInfo)
                        else:
                            edgeInfo = list()
                            edgeInfo.append(line.split(EDGE_FLAG)[0])
                            edgeInfo.append(line.split(EDGE_FLAG)[1][:-1])
                            edgeList.append(edgeInfo)

                # for i in edgeList:
                #     print(i)
                temp = edgeList[:]
                for edge in edgeList:
                    result = edge[:]
                    startPos = edge[0]
                    endPos = edge[1]
                    for line in temp:
                        if line[1] == startPos:
                            result.insert(0, line[0])
                            startPos = line[0]
                        if line[0] == endPos:
                            result.append(line[1])
                            endPos = line[1]
                    self.funcCallGraph.append(result)
                f.seek(0, 0)
                startFuncList = [funcName[0] for funcName in self.funcCallGraph]
                for line in f.readlines():
                    if line.find(LABEL_FLAG) != -1:
                        funcName = line.split(" ")[0]
                        if funcName not in startFuncList:
                            self.funcCallGraph.append([funcName])
                        else:
                            continue
            except:
                # print("Failed to read functions call-graph.")
                pass

    def getContractName(self):
        contract_ast = self.findASTNode(self.json, "name", "ContractDefinition")
        contractNameList = []
        for item in contract_ast:
            contractNameList.append(item["attributes"]["name"])
        return contractNameList


    def getCallValue(self, _json):
        memberList = self.findASTNode(_json,'name', 'MemberAccess')
        location = []
        for item in memberList:
            if item["attributes"]["member_name"] == "value" and item["children"][0]["name"] == "MemberAccess":
                if item["children"][0]["attributes"]["member_name"] == "call":
                    memStartPos,memEndPos = self.srcToPos(item['src'])
                    location.append([memStartPos,memEndPos])
        return location


    def getLocation(self, _json, _contractName, _functionName):
        result = []
        contractAST = self.findASTNode(_json, 'name', 'ContractDefinition')
        CVLocation = self.getCallValue(_json)
        for contractItem in contractAST:
            if contractItem['attributes']['name'] == _contractName:
                functionAST = self.findASTNode(contractItem, 'name', 'FunctionDefinition')
                for functionItem in functionAST:
                    if functionItem['attributes']['name'] == _functionName:
                        funcStartPos, funcEndPos = self.srcToPos(functionItem['src'])
                        for CVLitem in CVLocation:
                            if CVLitem[0] >= funcStartPos and CVLitem[1] <= funcEndPos:
                                result.append([funcStartPos, funcEndPos, FLAG_FUNC])
                            else:
                                pass
                    else:
                        pass
            else:
                pass
        return result

    def getOnePath(self, _json):
        pathList = []
        callPath = self.contractNameToNum(self.funcCallGraph)
        for oneCallPath in callPath:
            for onePathItem in oneCallPath:
                contractName = onePathItem.split('.')[0]
                funcName = onePathItem.split('.')[1]
                if self.getLocation(_json, contractName, funcName):
                    pathList.append(oneCallPath)
                else:
                    pass
        l = list(set([tuple(t) for t in pathList]))
        result = [list(v) for v in l]
        return result

    def getAddressVariable(self,_json,_contractName,_functionName):
        identifier_dict = {}
        elementList = []
        contract_ast = self.findASTNode(_json, "name", "ContractDefinition")
        for contract_item in contract_ast:
            if contract_item["attributes"]["name"] == _contractName:
                func_ast = self.findASTNode(contract_item, "name", "FunctionDefinition")
                for func_item in func_ast:
                    if func_item["attributes"]["name"] == _functionName:
                        member_ast = self.findASTNode(func_item, "name", "MemberAccess")
                        for member_item in member_ast:
                            if member_item["attributes"]["member_name"] == "call":
                                identifier_ast = self.findASTNode(member_item, "name", "Identifier")
                                eleExpression_ast = self.findASTNode(member_item, 'name', 'ElementaryTypeNameExpression')
                                mem_ast = self.findASTNode(member_item, 'name', 'MemberAccess')
                                for eleExpression_item in eleExpression_ast:
                                    if eleExpression_item['attributes']['value'] == 'address':
                                        elementList.append(eleExpression_item['attributes']['argumentTypes'][0]['typeString'])
                                for mem_item in mem_ast:
                                    if mem_item['attributes']['type'] == 'address':
                                        if mem_item['children'][0]['attributes']['referencedDeclaration'] and mem_item['attributes']['referencedDeclaration']:
                                            mem_declaration_ = mem_item['children'][0]['attributes']['referencedDeclaration']
                                            identifier_dict[mem_declaration_] = mem_item['attributes']['member_name']
                                for identifier_item in identifier_ast:
                                    if identifier_item["attributes"]["referencedDeclaration"]:
                                        if identifier_item["attributes"]["type"] == "address" or identifier_item['attributes']['type'] == 'address payable' or identifier_item["attributes"]["type"] == "contract OwnedUpgradeabilityProxy" or identifier_item["attributes"]["type"] == "address[] memory" :
                                            identifier_name = identifier_item["attributes"]["value"]
                                            identifier_id = identifier_item["attributes"]["referencedDeclaration"]
                                            identifier_dict[identifier_id] = identifier_name

                                        elif identifier_item['attributes']['type'] == 'msg':
                                            identifier_name = identifier_item['attributes']['value']
                                            identifier_id = identifier_item['attributes']['referencedDeclaration']
                                            identifier_dict[identifier_id] = identifier_name

                                        elif identifier_item['attributes']['type'] in elementList:
                                            identifier_name = identifier_item['attributes']['value']
                                            identifier_id = identifier_item['attributes']['referencedDeclaration']
                                            identifier_dict[identifier_id] = identifier_name
                                        else:
                                            continue
                                    else:
                                        identifier_name = identifier_item["attributes"]["value"]
                                        identifier_id = member_item["id"]
                                        identifier_dict[identifier_id] = identifier_name
        return identifier_dict


    def getAddressRelatedSC(self,_json,_contractName,_functionName):
        pos_list = []
        var_dict = self.getAddressVariable(_json, _contractName, _functionName)
        address_key_ = [key for key in var_dict.keys()][0]
        address_var_ = [var for var in var_dict.values()][0]
        addressID_ast = self.findASTNode(_json, 'id', address_key_)
        addressId_pos = []
        for addressID_item in addressID_ast:
            addressID_startPos,addressID_endPos = self.srcToPos(addressID_item['src'])
            addressId_pos.append([addressID_startPos,addressID_endPos])
        contract_ast = self.findASTNode(_json, 'name', 'ContractDefinition')
        for contractItem in contract_ast:
            if contractItem['attributes']['name'] == _contractName:
                contractStartPos,contractEndPos = self.srcToPos(contractItem['src'])
                func_ast = self.findASTNode(contractItem, "name", "FunctionDefinition")
                for funcItem in func_ast:
                    if funcItem['attributes']['name'] == _functionName:
                        funcStartPos,funcEndPos = self.srcToPos(funcItem['src'])
                        pos_list.append([funcStartPos, funcEndPos])
                        # 2.msg.sender
                        identifier_ast = self.findASTNode(funcItem, 'name', 'Identifier')
                        for identifierItem in identifier_ast:
                            if identifierItem['attributes']['referencedDeclaration'] == address_key_:
                                iden_startPos, _ = self.srcToPos(identifierItem['src'])
                                pos_list.append([iden_startPos])

                        for item in addressId_pos:
                            addressID_startPos = item[0]
                            addressID_endPos = item[1]
                            #
                            if addressID_startPos > funcStartPos  and addressID_endPos < funcEndPos:
                                identifier_ast = self.findASTNode(funcItem, "name", "Identifier")
                                for identifier_item in identifier_ast:
                                    if identifier_item["attributes"]["referencedDeclaration"] == address_key_:
                                        if identifier_item["attributes"]["type"] == "address" or \
                                                identifier_item["attributes"][
                                                    "type"] == "contract OwnedUpgradeabilityProxy" or \
                                                identifier_item["attributes"]["type"] == "address[] memory":
                                            identifier_startPos, _ = self.srcToPos(identifier_item["src"])
                                            pos_list.append([identifier_startPos])
                                        elif identifier_item['attributes']['type'] == 'msg':
                                            identifier_startPos,_ = self.srcToPos(identifier_item['src'])
                                            pos_list.append([identifier_startPos])
                                        else:
                                            continue
                                    else:
                                        continue
                            #
                            elif addressID_startPos > contractStartPos and addressID_endPos < contractEndPos:
                                pos_list.append([funcStartPos, funcEndPos])
                                identifier_ast_ = self.findASTNode(funcItem, "name", "Identifier")
                                for identifier_item_ in identifier_ast_:
                                    if identifier_item_["attributes"]["referencedDeclaration"] == address_key_:
                                        if identifier_item_["attributes"]["type"] == "address" or \
                                                identifier_item_["attributes"][
                                                    "type"] == "contract OwnedUpgradeabilityProxy" or \
                                                identifier_item_["attributes"]["type"] == "address[] memory":
                                            identifier_startPos_, _ = self.srcToPos(identifier_item_["src"])
                                            pos_list.append([identifier_startPos_])
                                        elif identifier_item_['attributes']['type'] == 'msg':
                                            identifier_startPos_, _ = self.srcToPos(identifier_item_['src'])
                                            pos_list.append([identifier_startPos_])
                                        else:
                                            continue
                                    else:
                                        continue
                            else:
                                continue
        return pos_list


    def getCallValueRealtedSC(self,_json):
        sc_list = []
        allPath = self.getOnePath(_json)
        for onepath in allPath:
            for onepathItem in onepath:
                contractName = onepathItem.split('.')[0]
                funcName = onepathItem.split('.')[1]
                variable = self.getAddressVariable(_json, contractName, funcName)
                if len(variable) == 0:
                    pass
                else:
                    sc = self.getAddressRelatedSC(_json, contractName, funcName)
                    sc_list.append(sc)
        sc = list(set([m for i in sc_list for j in i for m in j]))
        return sc

    def run(self):
        self.getAllFuncCallGraph()
        self.getCallGraphDot()
        sc_list = self.getCallValueRealtedSC(self.json)
        line = self.getLine(sc_list, self.contractPath)
        self.storeCodeSclingSC(line,self.contractPath,self.filename)

    def srcToPos(self, _src):
        temp = _src.split(":")
        return int(temp[0]), int(temp[0]) + int(temp[1])
    def srcToFirstPos(self,_src):
        temp = _src.split(":")
        return int(temp[0])

    def findASTNode(self, _ast, _name, _value):
        queue = [_ast]
        result = list()
        literalList = list()
        while len(queue) > 0:
            data = queue.pop()
            for key in data:
                if key == _name and data[key] == _value:
                    result.append(data)
                elif type(data[key]) == dict:
                    queue.append(data[key])
                elif type(data[key]) == list:
                    for item in data[key]:
                        if type(item) == dict:
                            queue.append(item)
        return result

    def splitTemp(self, _str):
        result = list()
        flag = 0
        temp = str()
        for char in _str:
            if char != "_":
                temp += char
            elif char == "_" and flag < 1:
                temp = str()
                flag += 1
            elif char == "_" and flag == 1:
                result.append(temp)
                temp = str()
                flag += 1
            elif flag >= 2:
                temp += char
        result.append(temp)
        return result[0], result[1]

    def splitTempName(self, _str):
        result = list()
        flag = False
        temp = str()
        for char in _str:
            if char == "_" and flag == False:
                flag = True
                result.append(temp)
                temp = ""
            else:
                temp += char
        result.append(temp)
        return result[0][1:], result[1][:-1]  #

    def getSourceCode(self, _path):
        try:
            with open(_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            raise Exception("Failed to get source code when detecting.")
            return str()

    def storeCodeSclingSC(self,address_relatedList,sc_infilepath,_preName):
        file = open(sc_infilepath, 'rb')
        data = file.readlines()
        with open(os.path.join(DATASET_PATH,_preName + EXTRACTED_CONTRACT_SUFFIX),'wb+') as f:
            for i in address_relatedList:
                f.write(data[i-1])


    def getLine(self,list_,sc_filepath):
        lineBreak = '\n'
        code = self.getSourceCode(sc_filepath)
        code_list_ = []
        for i in list_:
            code_list_.append(code[:i].count(lineBreak) + 1)
        return sorted(set(code_list_))

# if __name__ == '__main__':
#     _contractPath = './cache/temp.sol'
#     _json = './cache/temp.sol_json.ast'
#     with open(_json,'rb') as file:
#         json_result = json.load(file)
#     _filename = 'temp'
#     reen = extractReen(_contractPath, json_result, _filename)
#     reen.run()










