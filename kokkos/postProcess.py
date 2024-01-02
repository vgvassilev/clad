import getopt, sys

def replaceKokkosInlineFunction(stringIn):
    string_old = 'inline'
    string_new = 'KOKKOS_INLINE_FUNCTION'
    return stringIn.replace(string_old, string_new)

def useAutoInSubview(stringIn):
    if stringIn.find('Kokkos::subview') == -1 and stringIn.find('Kokkos::create_mirror_view') == -1:
        return stringIn
    index_eq = stringIn.find('=') - 1
    for index in range(1, index_eq):
        if stringIn[index_eq-index] == ' ':
            index1 = index_eq-index
            break
    index0 = 0
    for index in range(0, index_eq):
        if stringIn[index] != ' ':
            index0 = index
            break
    stringOut = stringIn[0:index0] + 'auto ' + stringIn[index1+1:-1] + stringIn[-1]
    return stringOut

def useKokkosNamespace(stringIn):
    return stringIn.replace(' ALL', ' Kokkos::ALL').replace('<View', '<Kokkos::View').replace(' View', ' Kokkos::View')

def getFunctionLineIDs(linesIn, fucntionName):
    index0 = -1
    index1 = 0
    for line in linesIn:
        if index0 == -1 and line[0] != ' '  and line.find(fucntionName) != -1:
            index0 = index1
        if index0 != -1 and line[0] == '}':
            break
        index1 += 1
    return index0, index1

def getVariableDeclLineID(linesIn, variableName, index0, index1):
    for index in range(index0, index1):
        if linesIn[index].find(' ' + variableName + ' =') != -1:
            return index
    return 0


def swapLinesForVariableDecl(linesIn, fucntionName, variableName, index0=-1, index1=-1):
    if index0 == -1 or index1 == -1:
        index0, index1 = getFunctionLineIDs(linesIn, fucntionName)
    indexVar = getVariableDeclLineID(linesIn, variableName, index0, index1)
    tmpLine = linesIn[indexVar]
    for index in range(0, indexVar-index0):
        linesIn[indexVar-index] = linesIn[indexVar-index-1]
    linesIn[index0+1] = tmpLine


def getType(linesIn, fucntionName, variableName, index0=-1, index1=-1):
    if index0 == -1 or index1 == -1:
        index0, index1 = getFunctionLineIDs(linesIn, fucntionName)
    index_tmp0 = linesIn[index0].find(' ' + variableName + ',')
    index_tmp1 = linesIn[index0].find(' ' + variableName + ')')
    if index_tmp0 != -1:
        index_end = index_tmp0
    elif index_tmp1 != -1:
        index_end = index_tmp1
    else:
        return 'none'
    bracket_lvl = 0
    for index in range(1, index_end):
        if bracket_lvl == 0 and linesIn[index0][index_end-index] == ',':
            index_begin = index_end-index+1
            break
        if bracket_lvl == 0 and linesIn[index0][index_end-index] == '(':
            index_begin = index_end-index+1
            break
        if linesIn[index0][index_end-index] == '>':
            bracket_lvl += 1
        if linesIn[index0][index_end-index] == '<':
            bracket_lvl -= 1
    return linesIn[index0][index_begin:index_end]


def swapTypeForTemplate(linesIn, fucntionName, variableName, index0=-1, index1=-1):
    if index0 == -1 or index1 == -1:
        index0, index1 = getFunctionLineIDs(linesIn, fucntionName)
    typeVar = getType(linesIn, fucntionName, variableName)
    template = 'type_' + variableName
    linesIn[index0] = 'template <typename ' + template + '> \n' + linesIn[index0]

    for index in range(index0, index1):
        linesIn[index] = linesIn[index].replace(typeVar, template)

    # Get the _d_ names and replace the clad::array_ref<Kokkos::view> by Kokkos::view directly.
    derivativeVarNames = []
    while linesIn[index0].find('clad::array_ref<' + template + ' >') != -1:
        indexVarName0 = linesIn[index0].find('clad::array_ref<' + template + ' >') + len('clad::array_ref<' + template + ' >') + 1
        for indexVarName in range(indexVarName0, len(linesIn[index0])):
            if linesIn[index0][indexVarName] == ',':
                indexVarName1 = indexVarName
                break
            if linesIn[index0][indexVarName] == ')':
                indexVarName1 = indexVarName
                break
        derivativeVarNames.append(linesIn[index0][indexVarName0:indexVarName1])
        linesIn[index0] = linesIn[index0].replace('clad::array_ref<' + template + ' >', template)
    for index in range(index0, index1):
        for derivativeVarName in derivativeVarNames:
            linesIn[index] = linesIn[index].replace('(* ' + derivativeVarName + ')', derivativeVarName)


def transform(filenameIn, filenameOut):

    fileIn = open(filenameIn, "r")
    linesIn = fileIn.readlines()
    fileOut = open(filenameOut, "w")

    swapLinesForVariableDecl(linesIn, 'f_grad', 'N1')

    for i in range(0, len(linesIn)):
        linesIn[i] = replaceKokkosInlineFunction(linesIn[i])
        linesIn[i] = useAutoInSubview(linesIn[i])
        linesIn[i] = useKokkosNamespace(linesIn[i])

    swapTypeForTemplate(linesIn, 'f_view_grad', 'a')

    for line in linesIn:
        fileOut.write(line)
    fileIn.close()
    fileOut.close()

argumentList = sys.argv[1:]
 
options = "hi:o:"
 
long_options = ["help", "filenameIn=", "filenameOut="]


filenameIn = ''
filenameOut = ''

try:
    arguments, values = getopt.getopt(argumentList, options, long_options)

    for currentArgument, currentValue in arguments:
 
        if currentArgument in ("-h", "--help"):
            print ("Displaying Help")
 
        elif currentArgument in ("-i", "--filenameIn"):
            filenameIn = currentValue
             
        elif currentArgument in ("-o", "--filenameOut"):
            filenameOut = currentValue

except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

if filenameIn != '' and filenameOut != '' :
    transform(filenameIn, filenameOut)
else:
    print("Missing arguments")
