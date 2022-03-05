# %%
import psycopg2
# %%
#connect to db
con = psycopg2.connect(
        host = "localhost",
        database =  "example_db",
        user = "postgres",
        password = "020300",
        port = 5432 
    )

#cursor
cur = con.cursor()

#execute query
cur.execute("CREATE TABLE test_table (title varchar(30), content text);")
cur.execute("INSERT INTO test_table (title, content) VALUES (%s, %s)", ('hello', 'qwerttyfdas'))
#close.connection
con.close()

#


# %%
