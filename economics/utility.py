class Utility():

    @staticmethod
    def sort_together(to_sort, other):
        sorted_to_sort = sorted(to_sort)
        sorted_other = [x for _,x in sorted(zip(to_sort, other))]

        return sorted_to_sort, sorted_other