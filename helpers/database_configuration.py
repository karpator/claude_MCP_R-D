from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class DatabaseType(Enum):
    """Database types enumeration"""
    SOLAW = "solaw"


@dataclass
class DatabaseInfo:
    """Complete database information"""
    uuid: str
    display_name: str
    index_pattern: str
    
    def get_index_name(self, environment: str) -> str:
        """Generate full index name for environment"""
        return self.index_pattern.format(environment=environment)


class DatabaseConfiguration:
    """Centralized database configuration management"""
    
    # Central UUID and configuration registry
    DATABASES = {
        DatabaseType.SOLAW: DatabaseInfo(
            uuid="09eaa4a7-dbef-46c3-96a6-83e82251d664",
            display_name="Business-to-Business",
            index_pattern=f"pipeline_solaw_test"
        )
    }
    
    DEFAULT_DATABASE = DatabaseType.SOLAW
    ALL_DATABASES = list(DATABASES.keys())
    
    @classmethod
    def get_uuid(cls, db_type: DatabaseType) -> str:
        """Get UUID for database type"""
        return cls.DATABASES[db_type].uuid
    
    @classmethod
    def get_index_name(cls, db_type: DatabaseType, environment: str = None) -> str:
        """Get full index name for database type"""
        env = environment
        return cls.DATABASES[db_type].get_index_name(env)
    
    @classmethod
    def parse_database_type(cls, value: str) -> Optional[DatabaseType]:
        """Parse string to DatabaseType"""
        if not value:
            return None
        
        value_lower = value.lower().strip()
        for db_type in cls.ALL_DATABASES:
            if db_type.value == value_lower:
                return db_type
        return None
    
    @classmethod
    def determine_database_from_index(cls, index_name: str) -> DatabaseType:
        """Determine database type from index name"""
        if not index_name:
            return cls.DEFAULT_DATABASE
            
        index_lower = index_name.lower()
        for db_type, info in cls.DATABASES.items():
            pattern_base = info.index_pattern
            if pattern_base in index_lower or f"_{db_type.value}_" in index_lower:
                return db_type
                
        return cls.DEFAULT_DATABASE
    
    @classmethod
    def determine_database_from_gcs_uri(cls, gcs_uri: str) -> DatabaseType:
        """Determine database type from GCS URI using UUID"""
        if not gcs_uri:
            return cls.DEFAULT_DATABASE
            
        for db_type, info in cls.DATABASES.items():
            if info.uuid in gcs_uri:
                return db_type
                
        return cls.DEFAULT_DATABASE
    
    @classmethod
    def validate_filter(cls, filter_list: Optional[List[str]]) -> List[DatabaseType]:
        """Validate and convert filter list to DatabaseType list"""
        if not filter_list:
            return cls.ALL_DATABASES
        
        valid_types = []
        for item in filter_list:
            db_type = cls.parse_database_type(item)
            if db_type and db_type not in valid_types:
                valid_types.append(db_type)
        
        return valid_types if valid_types else cls.ALL_DATABASES
    
    @classmethod
    def create_gcs_path(cls, db_type: DatabaseType, file_name: str, 
                       page_number: Optional[str] = None, environment: str = None) -> str:
        """Create standardized GCS path"""
        env = environment
        uuid = cls.get_uuid(db_type)
        
        if page_number:
            return f"gs://pluto-ai-admin-{env}/{uuid}/processed_backup/{file_name}_page_{page_number}.pdf"
        else:
            return f"gs://pluto-ai-admin-{env}/{uuid}/processed_backup/{file_name}.pdf"
    
    @classmethod
    def get_available_database_names(cls) -> List[str]:
        """Get list of available database type names"""
        return [db_type.value for db_type in cls.ALL_DATABASES]
    
    @classmethod
    def is_filter_valid(cls, database_filter: Optional[List[str]]) -> bool:
        """Check if database filter contains valid values only"""
        if not database_filter:
            return True  # Empty filter is valid (means use all)
        
        available_dbs = cls.get_available_database_names()
        return all(db.lower() in available_dbs for db in database_filter if db)
    
    @classmethod
    def get_uuid_by_name(cls, database_name: str) -> Optional[str]:
        """Get UUID by database name"""
        db_type = cls.parse_database_type(database_name)
        return cls.get_uuid(db_type) if db_type else None
    
    @classmethod
    def get_display_name(cls, db_type: DatabaseType) -> str:
        """Get display name for database type"""
        return cls.DATABASES[db_type].display_name