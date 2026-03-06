from typing import Dict, List


class Dataset(list):
    @property
    def records(self) -> List[Dict]:
        return list(self)
