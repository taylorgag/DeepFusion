
import os
import re
import subprocess  # 使用subprocess的popen，该popen是同步IO的
import json
from shutil import rmtree
import time  # 计时模块
from ree.progmaSlicing_code import extractReen



SOURCE_CODE_PATH = './sourceCode'
SOURCE_CODE_PREFIX_PATH = "./sourceCode"

RESULT_PATH = "./result/"


CACHE_PATH = "./cache/"


class reentrancyCodeSlicingExtractor:
    def __init__(self):
        self.cacheContractPath = "./cache/temp.sol"
        self.cacheJsonAstPath = "./cache/"
        self.cacheJsonAstName = "temp.sol_json.ast"
        self.defaultSolc = "0.5.0"
        self.maxSolc = "0.7.1"
        self.minSolc = "0.4.0"
        self.index = 0
        self.maxIndex = 100
        '''
        #try:
        compileResult = subprocess.run("ulimit -s 102400", check = True, shell = True)	

        except:
            print("Change stack size..failed")
        '''
        try:
            os.mkdir(CACHE_PATH)
        except:
            # print("The cache_ folder already exists.")
            pass

    def preFilter(self, _sourceCode):
        unsupportedPattern = re.compile(r"(\b)pragma(\s)+solidity(\s)+0(\.)4(\.)")
        if unsupportedPattern.search(self.cleanComment(_sourceCode)):
            return False
        return True

    def inStandardVersion(self, _nowVersion):
        standardList = ["0.4.15","0.4.16","0.4.17","0.4.18","0.4.19","0.4.20","0.4.21","0.4.22","0.4.23","0.4.24","0.4.25","0.4.26","0.5.0","0.5.1", "0.5.2", "0.5.3", "0.5.4", "0.5.5", "0.5.6", "0.5.7", "0.5.8", "0.5.9", "0.5.10","0.5.11", "0.5.12", "0.5.13", "0.5.14", "0.5.15", "0.5.16", "0.5.17","0.6.0", "0.6.1", "0.6.2", "0.6.3", "0.6.4", "0.6.5", "0.6.6", "0.6.7", "0.6.8", "0.6.9","0.6.10", "0.6.11", "0.6.12", "0.７.0", "0.7.1"]
        return _nowVersion in standardList

    def cleanComment(self, _code):

        singleLinePattern = re.compile(r"(//)(.)+")
        multipleLinePattern = re.compile(r"(/\*)(.)+?(\*/)")

        indexList = list()
        for item in singleLinePattern.finditer(_code):
            indexList.append(item.span())
        for item in multipleLinePattern.finditer(_code, re.S):
            indexList.append(item.span())
        startIndedx = 0
        newCode = str()
        for item in indexList:
            newCode += _code[startIndedx: item[0]]
            startIndedx = item[1] + 1
        newCode += _code[startIndedx:]
        return newCode

    def extractContracts(self):
        startTime = time.time()
        contractNum = 0
        while self.index < self.maxIndex:
            contractNum += 1
            try:
                (sourceCode, prevFileName) = self.getSourceCode()
                print("\r\t   Extracting contract: ", prevFileName)
                self.cacheContract(sourceCode)
                cache_file = os.listdir(self.cacheJsonAstPath)
                cache_num = 0
                self.changeSolcVersion(sourceCode)
                jsonAst = self.compile2Json()
                print(os.path.join(self.cacheContractPath))
                print(prevFileName)
                reen = extractReen(os.path.join(self.cacheContractPath), jsonAst, prevFileName)
                reen.run()
                rmtree(CACHE_PATH)
                os.mkdir(CACHE_PATH)
            except Exception as e:
                continue
        endTime = time.time()
        print()
        print(contractNum)
        '''
        if self.nowNum >= self.needs:
            print("Complete the extraction.")
        if self.index >= self.maxIndex:
            print("The data set lacks a sufficient number of contracts that meet the extraction criteria.")
        return
        '''

    def getSourceCode(self):
        fileList = os.listdir(SOURCE_CODE_PATH)
        solList = list()
        for i in fileList:
            if i.split(".")[1] == "sol":
                solList.append(i)
            else:
                continue
        self.maxIndex = len(solList)
        # index = randint(0, len(solList) - 1)
        index = self.index
        # print(index, solList[index])
        try:
            sourceCode = open(os.path.join(SOURCE_CODE_PREFIX_PATH, solList[index]), "r", encoding="utf-8").read()
            sourceCode = self.cleanMultibyte(sourceCode)
            self.index += 1
            # sourceCode = open(os.path.join(RESULT_PATH, solList[index]), "r", encoding = "utf-8").read()
            return sourceCode, solList[index]
        except:
            self.index += 1
            raise Exception("Unable to obtain source code " + solList[index])

    def cleanMultibyte(self, _sourceCode):
        result = str()
        for char in _sourceCode:
            if len(char) == len(char.encode()):
                result += char
            else:
                result += "1"
        return result

    def changeSolcVersion(self, _sourceCode):
        pragmaPattern = re.compile(r"(\b)pragma(\s)+(solidity)(\s)*(.)+?(;)")
        lowVersionPattern = re.compile(r"(\b)(\d)(\.)(\d)(.)(\d)+(\b)")
        pragmaStatement_mulLine = pragmaPattern.search(_sourceCode, re.S)
        pragmaStatement_sinLine = pragmaPattern.search(_sourceCode)
        pragmaStatement = pragmaStatement_sinLine if pragmaStatement_sinLine else pragmaStatement_mulLine

        if pragmaStatement:

            solcVersion = lowVersionPattern.search(pragmaStatement.group())
            print("solcVersion", solcVersion)
            if solcVersion:
                self.defaultSolc = solcVersion.group()
        print(self.inStandardVersion(self.defaultSolc))
        try:
            if self.inStandardVersion(self.defaultSolc):
                print("*********************" + self.defaultSolc + "****************************")
                compileResult = subprocess.run("solc use " + self.defaultSolc, check=True, shell=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:

                raise Exception("Use unsupported solc version.")
        except Exception as e:
            raise Exception("Failed to switch the solc version.")
            return
        # sys.exit(0)

    def cacheContract(self, _sourceCode):
        try:
            with open(self.cacheContractPath, "w+", encoding="utf-8") as f:
                f.write(_sourceCode)
            return
        except:
            raise Exception("Failed to cache_ contract.")

    def  compile2Json(self):
        try:
            subprocess.run("solc --ast-json --overwrite " + self.cacheContractPath + " -o " + self.cacheJsonAstPath,
                           check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open(self.cacheJsonAstPath + self.cacheJsonAstName, "r", encoding="utf-8") as f:
                compileResult = f.read()
            return json.loads(compileResult)
        except:
            raise Exception("Failed to compile the contract.")



if __name__ == "__main__":
    startTime = time.time()
    ree = reentrancyCodeSlicingExtractor()
    ree.extractContracts()
    endTime = time.time()
    print(endTime-startTime)

