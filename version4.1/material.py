class Material:
    def __init__(self, name, category, weight_per_unit,quantity,anxiety_factor):
        self.name = name
        self.category = category
        self.weight_per_unit = weight_per_unit
        self.quantity = quantity
        self.anxiety_factor = anxiety_factor

    def __str__(self):
        return self.name + " quantity: " + str(self.quantity)

    # 两方减，影响双方的数据，返回是否足够
    def __sub__(self, other):

        if isinstance(other, Material):
            is_enough = True if self.quantity >= other.quantity else False
            least = min(self.quantity,other.quantity)
            self.quantity -= least
            other.quantity -= least
            return is_enough
        else:
            is_enough = True if self.quantity >= other else False
            least = min(self.quantity,other)
            self.quantity -= least
            return is_enough

    # 影响原本的对象数据，返回当前对象
    def __add__(self, other):

        if isinstance(other, Material):
            self.quantity += other.quantity
            return self
        else:
            self.quantity += other
            return self



    def __lt__(self, other): #重载 self<record 运算符
        if isinstance(other, Material):
            if self.quantity < other.quantity:
                return True
            else:
                return False
        else:
            if self.quantity < other:
                return True
            else:
                return False


    def __int__(self):
        return int(self.quantity)

    def __float__(self):
        return float(self.quantity)


    def cal_anxiety_rate(self):
        return self.anxiety_factor * self.quantity



class AMaterial(Material):
    def __init__(self, quantity):
        super().__init__("EmergencyMaterial", "A", 1,quantity,0.5)

class BMaterial(Material):
    def __init__(self, quantity):
        super().__init__("RegularMaterial", "B", 3,quantity,0.3)

class CMaterial(Material):
    def __init__(self, quantity):
        super().__init__("EquipmentMaterial", "C", 10,quantity,0.1)


class MaterialPackage:
    def __init__(self, A_num, B_num, C_num):
        self.A_material = AMaterial(A_num)
        self.B_material = BMaterial(B_num)
        self.C_material = CMaterial(C_num)

    def __sub__(self, other):

        is_A_enough = self.A_material - other.A_material
        is_B_enough = self.B_material - other.B_material
        is_C_enough = self.C_material - other.C_material

        is_enoungh = is_A_enough and is_B_enough and is_C_enough

        return is_enoungh

    def __add__(self, other):
        self.A_material = self.A_material + other.A_material
        self.B_material = self.B_material + other.B_material
        self.C_material = self.C_material + other.C_material
        return self

    def __str__(self):
        return "package:\n\t" + str(self.A_material)  + "\n\t"+ str(self.B_material) + "\n\t" + str(self.C_material)

    def cal_anxiety_rate(self):
        rate = float(self.A_material.cal_anxiety_rate()) + (self.B_material.cal_anxiety_rate()) + (self.C_material.cal_anxiety_rate())
        return rate


if __name__ == '__main__':
    need = MaterialPackage(20 , 10,2)
    package = MaterialPackage(30,5,1)
    print(need)
    print(package)

    isenough =  package - need

    print(need)
    print(package)
    print(isenough)

# class Material:
#     def __init__(self, name, category, weight_per_unit , quantity = 0 ,anxiety_factor=0):
#         self.name = name
#         self.category = category
#         self.weight_per_unit = weight_per_unit
#         self.anxiety_factor = anxiety_factor
#         self.quantity = quantity # 补给点为数量、受灾点为需求
#
#     def cal_anxiety_rate(self):
#         return self.anxiety_factor * self.quantity
#
#
# class MaterialPackage:
#     def __init__(self,A_material,B_material,C_material):
#         self.A_material = A_material
#         self.B_material = B_material
#         self.C_material = C_material
#
#     def sub(self,package):
#         is_enoungh = True
#         self.A_material -=